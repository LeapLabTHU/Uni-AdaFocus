import torch.nn.parallel
import torch.optim

from ops.transforms import *

from archs.x3d_video_model_builder import X3D
from slowfast.config.defaults import get_cfg
import slowfast.utils.checkpoint as cu


import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast

best_prec1 = 0


def get_patch_grid_scalexy(input_frames, action, image_size, patch_size, input_patch_size):
    batchsize = action.size(0)
    theta = torch.zeros((batchsize, 2, 3), device=input_frames.device)
    patch_scale = action[:, 2:4] * (224 - 96) / input_patch_size + 96 / input_patch_size
    patch_coordinate = (action[:, :2] * (image_size - patch_size * patch_scale))
    x1, x2, y1, y2 = patch_coordinate[:, 1], patch_coordinate[:, 1] + patch_size * patch_scale[:, 1], \
                     patch_coordinate[:, 0], patch_coordinate[:, 0] + patch_size * patch_scale[:, 0]

    theta[:, 0, 0], theta[:, 1, 1] = patch_size * patch_scale[:, 1] / image_size, patch_size * patch_scale[:, 0] / image_size
    theta[:, 0, 2], theta[:, 1, 2] = -1 + (x1 + x2) / image_size, -1 + (y1 + y2) / image_size

    grid = F.affine_grid(theta.float(), torch.Size((batchsize, 3, patch_size, patch_size)))

    patches = F.grid_sample(input_frames, grid)

    return patches


def cal_x(x1, y1, x2, y2, y):
    x = (x2 * (y - y1) - x1 * (y - y2)) / (y2 - y1)
    return x


def policy_sample_indices(weights, num_segments, num_global_segments, num_local_segments, rand_sample=True, device="cpu"):
    _b, _ = weights.shape
    integrated = weights.cumsum(dim=1)
    new_col = torch.zeros(_b, device=device)
    integrated = torch.cat([new_col.unsqueeze(1), integrated], dim=1)  # Accumulation of weights, with leading 0
    integrated[:, -1] = 1.0  # fix precision loss
    threshold = torch.arange(0, num_local_segments, device=device).float().repeat(_b, 1)
    if rand_sample:
        threshold = (threshold + torch.rand_like(threshold)) / num_local_segments
    else:
        threshold = (threshold + 0.5) / num_local_segments
    # find first integrated[j, ans] > threshold[j, i] - eps
    cmp_integrated_threshold = integrated.unsqueeze(2) >= threshold.unsqueeze(1)
    first_cur = cmp_integrated_threshold.max(dim=1)[1] - 1
    quantile = cal_x(x1=first_cur / num_global_segments, y1=integrated.gather(1, first_cur),
                     x2=(first_cur + 1) / num_global_segments, y2=integrated.gather(1, first_cur + 1),
                     y=threshold)
    # round float to integer index
    real_indices = quantile * num_segments
    int_indices = torch.floor(real_indices).long()
    # fix overlap situation
    for i in range(1, num_local_segments):
        int_indices[:, i] = torch.where(int_indices[:, i].gt(int_indices[:, i - 1]), int_indices[:, i], int_indices[:, i - 1] + 1)
    lim = num_segments - num_local_segments + torch.arange(0, num_local_segments, device=device)
    lim = lim.unsqueeze(0)
    int_indices = torch.where(int_indices.gt(lim), lim, int_indices)
    offset = torch.arange(0, _b, device=device).unsqueeze(1) * num_segments

    int_indices = int_indices + offset
    return int_indices.view(-1)


def MCSampleFeature(global_feat, weights, global_feature_dim, T1, sample_times=1000, device="cpu"):
    B, _ = weights.shape
    global_feat = global_feat.view(B, -1, global_feature_dim)
    _, T, Dim = global_feat.shape
    all_weights = weights
    all_weights = all_weights.unsqueeze(1).repeat(1, sample_times, 1).view(-1, T)
    sum_feat = torch.zeros([B * sample_times, global_feature_dim], device=device).float()
    for i in range(T1):
        new_weights = all_weights.clone()
        if i > 0:
            indices = torch.multinomial(all_weights, i, replacement=False)
            new_weights = new_weights.scatter(dim=1, index=indices, value=0.0)
        new_feat = global_feat.unsqueeze(1).repeat(1, sample_times, 1, 1).view(-1, T, Dim)
        frame_feat = (new_feat * new_weights.unsqueeze(-1)).sum(dim=1)
        frame_feat = frame_feat / new_weights.sum(dim=1).detach().unsqueeze(-1)
        sum_feat = sum_feat + frame_feat
    sum_feat = sum_feat / T1
    sum_feat = sum_feat.view(-1, sample_times, Dim).mean(dim=1)
    return sum_feat


class TemporalPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_glance_segments, temperature):
        super().__init__()
        self.input_dim = input_dim
        self.num_glance_segments = num_glance_segments
        self.T = temperature
        self.Tconv = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.input_dim,
                out_channels=hidden_dim,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            )
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, self.num_glance_segments, self.input_dim).permute(0, 2, 1).contiguous().unsqueeze(-1)
        x = self.Tconv(x).flatten(1)
        x1 = self.softmax(x)
        x2 = x / self.T
        x2 = self.softmax(x2)
        return x1, x2


class SpatialPolicy(nn.Module):
    def __init__(self, stn_feature_dim, hidden_dim, num_glance_segments):
        super(SpatialPolicy, self).__init__()
        self.stn_feature_dim = stn_feature_dim
        self.num_glance_segments = num_glance_segments
        self.encoder = nn.Sequential(
            nn.Conv3d(
                stn_feature_dim, hidden_dim,
                kernel_size=(1, 1, 1), stride=1, padding=0, bias=False,
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(
                hidden_dim, hidden_dim,
                kernel_size=(3, 3, 3), stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(
                hidden_dim, hidden_dim,
                kernel_size=(3, 3, 3), stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(
                hidden_dim, 4,
                kernel_size=(self.num_glance_segments, 5, 5), stride=1, padding=(0, 0, 0), bias=False,
            ),
            nn.Sigmoid()
        )

    def forward(self, features):
        N, TC, H, W = features.shape
        features = features.reshape(
            N, self.num_glance_segments, self.stn_feature_dim, H, W
        ).permute(0, 2, 1, 3, 4).contiguous()
        actions = self.encoder(features).reshape(N, 4)
        return actions


def load_config(path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)

    return cfg


class AdaFocus(nn.Module):
    def __init__(self, num_class, args):
        super(AdaFocus, self).__init__()
        self.num_glance_segments = args.num_glance_segments
        self.num_input_focus_segments = args.num_input_focus_segments
        self.num_focus_segments = args.num_focus_segments
        self.input_size = args.input_size
        self.glance_size = args.glance_size
        self.patch_size = args.patch_size
        self.num_classes = num_class
        self.device = args.device

        global_cfg = load_config('configs/X3D_S.yaml')
        global_cfg.DATA.NUM_FRAMES = self.num_glance_segments
        global_cfg.MODEL.NUM_CLASSES = self.num_classes
        global_cfg.DATA.TRAIN_CROP_SIZE = self.glance_size  # 160
        self.global_CNN = X3D(cfg=global_cfg)
        self.global_feature_dim = self.global_CNN.head.lin_5.in_channels
        self.global_final_dim = self.global_CNN.head.projection.in_features
        self.global_feat_map_size = 5
        self.global_feat_map_patch_size = round(self.global_feat_map_size * self.patch_size / self.input_size)
        self.global_CNN_head = nn.Sequential(
            nn.Linear(self.global_feature_dim, self.global_final_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(self.global_final_dim, self.num_classes, bias=True),
        )
        local_cfg = load_config('configs/X3D_L.yaml')
        local_cfg.DATA.NUM_FRAMES = self.num_focus_segments
        local_cfg.MODEL.NUM_CLASSES = self.num_classes
        local_cfg.DATA.TRAIN_CROP_SIZE = self.patch_size
        self.local_CNN = X3D(cfg=local_cfg)
        self.local_feature_dim = self.local_CNN.head.lin_5.in_channels
        self.local_final_dim = self.local_CNN.head.projection.in_features
        self.local_CNN_head = nn.Sequential(
            nn.Linear(self.local_feature_dim, self.local_final_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(self.local_final_dim, self.num_classes),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(self.global_feature_dim, self.num_classes)
        )
        self.spatial_policy = SpatialPolicy(
            stn_feature_dim=self.global_feature_dim,
            hidden_dim=args.stn_hidden_dim,
            num_glance_segments=self.num_glance_segments
            )
        self.temporal_policy = TemporalPolicy(
            input_dim=self.global_feature_dim,
            hidden_dim=args.temporal_hidden_dim,
            num_glance_segments=self.num_glance_segments,
            temperature=1.0
        )

        print(f'Loading global_CNN: X3D-S...')
        self.load_pretrain(self.global_CNN, global_cfg, 'archs/X3D_ckpt/x3d_s.pyth')
        print(f'Loading local_CNN: X3D-L...')
        self.load_pretrain(self.local_CNN, local_cfg, 'archs/X3D_ckpt/x3d_l.pyth')
        print("glance_size:", self.glance_size)
        print("global_feat_map_size:", self.global_feat_map_size)
        print("global_feat_map_patch_size:", self.global_feat_map_patch_size)
        print("patch_size:", self.patch_size)
        self.share_head_params()

    def forward(self, images_glance, images_input):
        # Perform global CNN
        images_glance = F.interpolate(images_glance, (self.glance_size, self.glance_size),
                                      mode='bilinear')
        images_glance = images_glance.view(-1, self.num_glance_segments, 3, self.glance_size, self.glance_size)
        images_glance = images_glance.permute([0, 2, 1, 3, 4]).contiguous()
        with autocast(enabled=self.training):
            global_final_logit, global_feat_maps, global_feat = self.global_CNN.get_featmap(images_glance)
        global_final_logit = global_final_logit.float()
        global_feat_maps = global_feat_maps.float()
        global_feat = global_feat.float()
        global_feat_maps = global_feat_maps.permute([0, 2, 1, 3, 4]).contiguous() \
            .view(-1, self.num_glance_segments * self.global_feature_dim, 5, 5)
        global_feat = global_feat.permute([0, 2, 1]).contiguous().view(-1, self.global_feature_dim)
        # Perform global CNN mlp
        global_logits = self.global_CNN_head(global_feat)
        global_avg_logit = global_logits.view(-1, self.num_glance_segments, self.num_classes).mean(1)
        # Perform temporal policy
        weights, weights_T = self.temporal_policy(global_feat.detach())
        temporal_sample_logits = MCSampleFeature(
            global_logits.view(-1, self.num_classes).detach(),
            weights,
            global_feature_dim=self.num_classes,
            T1=round(self.num_glance_segments / 3.0),
            sample_times=128,
            device=self.device)
        # Perform spatial policy
        action_3 = self.spatial_policy(global_feat_maps.detach())
        patches_3 = get_patch_grid_scalexy(
            input_frames=global_feat_maps.detach(),
            action=action_3,
            image_size=self.global_feat_map_size,
            patch_size=self.global_feat_map_patch_size,
            input_patch_size=self.patch_size
        )
        aux_global_feat_3 = patches_3.view(-1, self.num_glance_segments, self.global_feature_dim,
                                           self.global_feat_map_patch_size, self.global_feat_map_patch_size)
        aux_global_feat_3 = aux_global_feat_3.mean(dim=[1, 3, 4])
        spatial_sample_logits_3 = self.aux_head(aux_global_feat_3)

        if self.training:
            # Perform temporal sample
            focus_indices = policy_sample_indices(weights_T.detach(), self.num_input_focus_segments,
                                                               self.num_glance_segments,
                                                               self.num_focus_segments, True, self.device)
            images_focus = images_input.view(-1, 3, self.input_size, self.input_size)[focus_indices]
            images_focus = images_focus.view(-1, self.num_focus_segments * 3, self.input_size, self.input_size)

            # Perform spatial policy
            action_2 = self.spatial_policy(global_feat_maps.detach().float())
            action_1 = torch.rand_like(action_2)
            action_1[:, 2:] = (self.patch_size - 96) / (224 - 96)
            patches_1 = get_patch_grid_scalexy(
                input_frames=images_focus,
                action=action_1.detach(),
                image_size=images_focus.size(2),
                patch_size=self.patch_size,
                input_patch_size=self.patch_size
            )
            patches_2 = get_patch_grid_scalexy(
                input_frames=images_focus,
                action=action_2.detach(),
                image_size=images_focus.size(2),
                patch_size=self.patch_size,
                input_patch_size=self.patch_size
            )
            # Perform local CNN
            local_input = torch.cat([patches_1, patches_2], dim=0)
            local_input = local_input.view(-1, self.num_focus_segments, 3, self.patch_size, self.patch_size).permute(
                [0, 2, 1, 3, 4])
            with autocast(enabled=self.training):
                local_final_logits, local_feat_maps, local_feature = self.local_CNN.get_featmap(local_input)
            local_final_logits = local_final_logits.float()
            local_feat_maps = local_feat_maps.float()
            local_feature = local_feature.float()
            local_feature = local_feature.permute([0, 2, 1]).contiguous().view(-1, self.local_feature_dim)
            # Perform local CNN fc
            local_logits = self.local_CNN_head(local_feature)
            local_logits = local_logits.view(2, -1, self.num_classes)
            local_logits_1 = local_logits[0]
            local_logits_2 = local_logits[1]
            local_avg_logit_1 = local_logits_1.view(-1, self.num_focus_segments, self.num_classes).mean(1)
            local_avg_logit_2 = local_logits_2.view(-1, self.num_focus_segments, self.num_classes).mean(1)
            local_final_logits = local_final_logits.view(2, -1, self.num_classes)
            local_final_logit_1 = local_final_logits[0]
            local_final_logit_2 = local_final_logits[1]

            return (global_final_logit + local_final_logit_1, global_avg_logit, local_avg_logit_1, temporal_sample_logits, spatial_sample_logits_3, action_1, action_3), \
                   (global_final_logit + local_final_logit_2, global_avg_logit, local_avg_logit_2, temporal_sample_logits, spatial_sample_logits_3, action_2, action_3)

        else:
            # Perform temporal sample
            focus_indices = policy_sample_indices(weights_T.detach(), self.num_input_focus_segments,
                                                               self.num_glance_segments,
                                                               self.num_focus_segments, False, self.device)
            images_focus = images_input.view(-1, 3, self.input_size, self.input_size)[focus_indices]
            images_focus = images_focus.view(-1, self.num_focus_segments * 3, self.input_size, self.input_size)

            # Perform spatial policy
            action = self.spatial_policy(global_feat_maps.detach())
            patches = get_patch_grid_scalexy(
                input_frames=images_focus,
                action=action,
                image_size=images_focus.size(2),
                patch_size=self.patch_size,
                input_patch_size=self.patch_size
            )
            # Perform local CNN
            local_input = patches.view(-1, self.num_focus_segments, 3, self.patch_size, self.patch_size).permute(
                [0, 2, 1, 3, 4])
            local_final_logit, local_feat_map, local_feature = self.local_CNN.get_featmap(local_input)
            local_feature = local_feature.permute([0, 2, 1]).contiguous().view(-1, self.local_feature_dim)
            # Perform local CNN fc
            local_logit = self.local_CNN_head(local_feature)
            local_avg_logit = local_logit.view(-1, self.num_focus_segments, self.num_classes).mean(1)
            # TODO
            # This is only for early-exit usage, please change the following line when training
            return global_avg_logit, global_final_logit + local_final_logit, global_avg_logit, local_avg_logit, temporal_sample_logits, spatial_sample_logits_3, action, action_3

    def load_pretrain(self, model, cfg, ckpt_path):
        checkpoint_epoch = cu.load_checkpoint(
            ckpt_path,
            model,
            data_parallel=False,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
        )
        return checkpoint_epoch

    def share_head_params(self):
        print(f'Load parameters to headers...')
        source_head_list = [self.global_CNN.head, self.local_CNN.head]
        target_head_list = [self.global_CNN_head, self.local_CNN_head]
        for source_head, target_head in zip(source_head_list, target_head_list):
            param_0 = source_head.lin_5.state_dict()
            param_0["weight"] = param_0["weight"].squeeze()  # from conv3d(1, 1, 1) to linear
            target_head[0].load_state_dict(param_0)
            param_3 = source_head.projection.state_dict()
            target_head[3].load_state_dict(param_3)
        print(f'Loaded')
        pass

    def get_optim_policies(self, args):
        return [{'params': self.temporal_policy.parameters(), 'lr_mult': args.temporal_lr_ratio, 'decay_mult': 1, 'name': "temporal_policy"}] \
               + [{'params': self.spatial_policy.parameters(), 'lr_mult': args.stn_lr_ratio, 'decay_mult': 1, 'name': "spatial_policy"}] \
               + [{'params': self.global_CNN.parameters(), 'lr_mult': args.global_lr_ratio, 'decay_mult': 1, 'name': "global_CNN"}] \
               + [{'params': self.global_CNN_head.parameters(), 'lr_mult': args.global_lr_ratio, 'decay_mult': 1,
                   'name': "global_CNN_head"}] \
               + [{'params': self.aux_head.parameters(), 'lr_mult': args.global_lr_ratio, 'decay_mult': 1, 'name': "aux_head"}] \
               + [{'params': self.local_CNN.parameters(), 'lr_mult': 1, 'decay_mult': 1, 'name': "local_CNN"}] \
               + [{'params': self.local_CNN_head.parameters(), 'lr_mult': 1, 'decay_mult': 1, 'name': "local_CNN_head"}]

