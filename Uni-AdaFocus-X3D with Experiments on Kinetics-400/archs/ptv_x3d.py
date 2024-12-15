#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Code is adapted from https://github.com/facebookresearch/SlowFast/tree/main/slowfast/models


"""Video models using PyTorchVideo model builder."""

from functools import partial
import torch
import torch.nn as nn
from ops.transforms import *

from archs.x3d import (
    Swish,
    create_x3d,
    create_x3d_bottleneck_block,
)

from pytorchvideo.layers.batch_norm import (
    NaiveSyncBatchNorm1d,
    NaiveSyncBatchNorm3d,
)  # noqa

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "slow_c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow_i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

def get_head_act(act_func):
    """
    Return the actual head activation function given the activation fucntion name.

    Args:
        act_func (string): activation function to use. 'softmax': applies
        softmax on the output. 'sigmoid': applies sigmoid on the output.
    Returns:
        nn.Module: the activation layer.
    """
    if act_func == "softmax":
        return nn.Softmax(dim=1)
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError(
            "{} is not supported as a head activation "
            "function.".format(act_func)
        )

def get_norm(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the normalization layer.
    """
    if cfg.BN.NORM_TYPE in {"batchnorm", "sync_batchnorm_apex"}:
        return nn.BatchNorm3d
    elif cfg.BN.NORM_TYPE == "sub_batchnorm":
        return partial(SubBatchNorm3d, num_splits=cfg.BN.NUM_SPLITS)
    elif cfg.BN.NORM_TYPE == "sync_batchnorm":
        return partial(
            NaiveSyncBatchNorm3d,
            num_sync_devices=cfg.BN.NUM_SYNC_DEVICES,
            global_sync=cfg.BN.GLOBAL_SYNC,
        )
    else:
        raise NotImplementedError(
            "Norm type {} is not supported".format(cfg.BN.NORM_TYPE)
        )

class SubBatchNorm3d(nn.Module):
    """
    The standard BN layer computes stats across all examples in a GPU. In some
    cases it is desirable to compute stats across only a subset of examples
    (e.g., in multigrid training https://arxiv.org/abs/1912.00998).
    SubBatchNorm3d splits the batch dimension into N splits, and run BN on
    each of them separately (so that the stats are computed on each subset of
    examples (1/N of batch) independently. During evaluation, it aggregates
    the stats from all splits into one BN.
    """

    def __init__(self, num_splits, **args):
        """
        Args:
            num_splits (int): number of splits.
            args (list): other arguments.
        """
        super(SubBatchNorm3d, self).__init__()
        self.num_splits = num_splits
        num_features = args["num_features"]
        # Keep only one set of weight and bias.
        if args.get("affine", True):
            self.affine = True
            args["affine"] = False
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.affine = False
        self.bn = nn.BatchNorm3d(**args)
        args["num_features"] = num_features * num_splits
        self.split_bn = nn.BatchNorm3d(**args)

    def _get_aggregated_mean_std(self, means, stds, n):
        """
        Calculate the aggregated mean and stds.
        Args:
            means (tensor): mean values.
            stds (tensor): standard deviations.
            n (int): number of sets of means and stds.
        """
        mean = means.view(n, -1).sum(0) / n
        std = (
            stds.view(n, -1).sum(0) / n
            + ((means.view(n, -1) - mean) ** 2).view(n, -1).sum(0) / n
        )
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """
        Synchronize running_mean, and running_var. Call this before eval.
        """
        if self.split_bn.track_running_stats:
            (
                self.bn.running_mean.data,
                self.bn.running_var.data,
            ) = self._get_aggregated_mean_std(
                self.split_bn.running_mean,
                self.split_bn.running_var,
                self.num_splits,
            )

    def forward(self, x):
        if self.training:
            n, c, t, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x = x * self.weight.view((-1, 1, 1, 1))
            x = x + self.bias.view((-1, 1, 1, 1))
        return x

class PTVX3D(nn.Module):
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PTVX3D, self).__init__()

        assert (
            cfg.RESNET.STRIDE_1X1 is False
        ), "STRIDE_1x1 must be True for PTVX3D"
        assert (
            cfg.RESNET.TRANS_FUNC == "x3d_transform"
        ), f"Unsupported TRANS_FUNC type {cfg.RESNET.TRANS_FUNC} for PTVX3D"
        assert (
            cfg.DETECTION.ENABLE is False
        ), "Detection model is not supported for PTVX3D yet."

        self._construct_network(cfg)

        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def _construct_network(self, cfg):
        """
        Builds a X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # Params from configs.
        norm_module = get_norm(cfg)
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.model = create_x3d(
            # Input clip configs.
            input_channel=cfg.DATA.INPUT_CHANNEL_NUM[0],
            input_clip_length=cfg.DATA.NUM_FRAMES,
            input_crop_size=cfg.DATA.TRAIN_CROP_SIZE,
            # Model configs.
            model_num_class=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            width_factor=cfg.X3D.WIDTH_FACTOR,
            depth_factor=cfg.X3D.DEPTH_FACTOR,
            # Normalization configs.
            norm=norm_module,
            norm_eps=1e-5,
            norm_momentum=0.1,
            # Activation configs.
            activation=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            # Stem configs.
            stem_dim_in=cfg.X3D.DIM_C1,
            stem_conv_kernel_size=(temp_kernel[0][0][0], 3, 3),
            stem_conv_stride=(1, 2, 2),
            # Stage configs.
            stage_conv_kernel_size=(
                (temp_kernel[1][0][0], 3, 3),
                (temp_kernel[2][0][0], 3, 3),
                (temp_kernel[3][0][0], 3, 3),
                (temp_kernel[4][0][0], 3, 3),
            ),
            stage_spatial_stride=(2, 2, 2, 2),
            stage_temporal_stride=(1, 1, 1, 1),
            bottleneck=create_x3d_bottleneck_block,
            bottleneck_factor=cfg.X3D.BOTTLENECK_FACTOR,
            se_ratio=0.0625,
            inner_act=Swish,
            # Head configs.
            head_dim_out=cfg.X3D.DIM_C5,
            head_pool_act=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            head_bn_lin5_on=cfg.X3D.BN_LIN5,
            head_activation=None,
            head_output_with_global_average=False,
        )

        self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)

    def forward(self, x, bboxes=None):
        x = self.model(x)
        # Performs fully convlutional inference.
        if not self.training:
            x = self.post_act(x)
            x = x.mean([2, 3, 4])

        x = x.reshape(x.shape[0], -1)
        return x

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
