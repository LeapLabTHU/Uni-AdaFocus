import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from ops.transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip
from .resnet import resnet50
from .mobilenet_v2 import mobilenet_v2


def get_patch_grid_scalexy(input_frames, actions, patch_size, image_size, input_patch_size):
    batchsize = actions.size(0)
    theta = torch.zeros((batchsize, 2, 3), device=input_frames.device)
    patch_scale = actions[:, 2:4] * (224 - 96) / input_patch_size + 96 / input_patch_size
    patch_coordinate = (actions[:, :2] * (image_size - patch_size * patch_scale))
    x1, x2, y1, y2 = patch_coordinate[:, 1], patch_coordinate[:, 1] + patch_size * patch_scale[:, 1], \
                     patch_coordinate[:, 0], patch_coordinate[:, 0] + patch_size * patch_scale[:, 0]

    theta[:, 0, 0], theta[:, 1, 1] = patch_size * patch_scale[:, 1] / image_size, patch_size * patch_scale[:,
                                                                                               0] / image_size
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
    first_cur = cmp_integrated_threshold.max(dim=1)[1].clamp(1, num_global_segments) - 1
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


def MCSampleFeature(global_feat, weights, global_feature_dim, T1, sample_times=128, device="cpu"):
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


class AdaFocus(nn.Module):
    def __init__(self, args):
        super(AdaFocus, self).__init__()
        self.num_segments = args.num_segments
        self.num_glance_segments = args.num_glance_segments
        self.num_focus_segments = args.num_focus_segments
        self.num_classes = args.num_classes
        if args.dataset == 'fcvid':
            assert args.num_classes == 239
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.glance_arch = args.glance_arch
        self.device = args.device

        print('Global CNN Backbone: mobilenet_v2')
        self.global_CNN = mobilenet_v2(pretrained=True)
        self.global_feature_dim = self.global_CNN.last_channel
        self.global_CNN_fc = nn.Sequential(
            nn.Dropout(args.fc_dropout),
            nn.Linear(self.global_feature_dim, self.num_classes),
        )
        self.aux_fc = nn.Linear(self.global_feature_dim, self.num_classes)
        print('Local CNN Backbone: resnet_50')
        self.local_CNN = resnet50(pretrained=True)
        self.local_feature_dim = self.local_CNN.fc.in_features
        self.local_CNN_fc = nn.Sequential(
            nn.Dropout(args.fc_dropout),
            nn.Linear(self.local_feature_dim, self.num_classes),
        )

        self.spatial_policy = SpatialPolicy(
            stn_feature_dim=self.global_feature_dim,
            num_glance_segments=self.num_glance_segments,
            hidden_dim=args.stn_hidden_dim,
        )
        self.temporal_policy = TemporalPolicy(
            input_dim=self.global_feature_dim,
            hidden_dim=args.temporal_hidden_dim,
            num_glance_segments=self.num_glance_segments,
            temperature=args.temperature,
            device=self.device
        )
        self.classifier = PoolingClassifier(
            global_feature_dim=self.global_feature_dim,
            local_feature_dim=self.local_feature_dim,
            num_glance_segments=self.num_glance_segments,
            num_focus_segments=self.num_focus_segments,
            num_classes=args.num_classes,
            dropout=args.dropout,
            device=self.device
        )
        print('AdaFocus Network Built')

    def forward(self, images):
        _b = images.size(0)
        times = self.num_segments // self.num_glance_segments
        if self.training:  # segment-wise random select
            glance_indices = torch.randint(0, times, [_b, self.num_glance_segments]) + \
                             (torch.arange(0, self.num_glance_segments) * times).repeat(_b, 1)
        else:
            glance_indices = torch.arange(self.num_segments)[1::self.num_segments // self.num_glance_segments].repeat(_b, 1)
        glance_indices = glance_indices + torch.arange(0, _b).unsqueeze(1) * self.num_segments
        glance_indices = glance_indices.view(-1)
        images = images.view(-1, 3, self.input_size, self.input_size)
        # Perform global CNN
        glance_images = images[glance_indices]
        with autocast(enabled=self.training):
            global_feat_map, global_feat = self.global_CNN.get_featmap(glance_images)
        global_feat = global_feat.float()
        global_feat_map = global_feat_map.float()
        global_logits = self.global_CNN_fc(global_feat)
        # Perform spatial policy
        actions_1 = self.spatial_policy(global_feat_map.detach())
        actions_1 = actions_1.view(-1, self.num_glance_segments, 4).permute(0, 2, 1)
        actions_1 = F.interpolate(actions_1, self.num_segments, mode='linear', align_corners=False)
        actions_1 = actions_1.permute(0, 2, 1).contiguous().view(-1, 4)
        actions_2 = torch.rand_like(actions_1)
        actions_2[:, 2:] = (self.patch_size - 96) / (224 - 96)
        # Perform temporal policy
        weights, weights_T = self.temporal_policy(global_feat)
        # Calculate temporal logits with Monte Carlo
        temporal_sample_logits = MCSampleFeature(
            global_logits,
            weights,
            global_feature_dim=self.num_classes,
            T1=self.num_glance_segments // 3,
            sample_times=128,
            device=self.device)
        # Calculate spatial logits with feature map interpolation
        action_3 = self.spatial_policy(global_feat_map.detach())
        patches_3 = get_patch_grid_scalexy(
            input_frames=global_feat_map.detach(),
            actions=action_3,
            image_size=7,
            patch_size=self.patch_size // 32,
            input_patch_size=self.patch_size,
        )
        aux_global_feat_3 = patches_3.view(-1, self.num_glance_segments, self.global_feature_dim, self.patch_size // 32,
                                           self.patch_size // 32)
        aux_global_feat_3 = aux_global_feat_3.mean(dim=[1, 3, 4])
        spatial_sample_logits_3 = self.aux_fc(aux_global_feat_3)

        if self.training:
            res = []
            focus_indices = policy_sample_indices(weights_T.detach(), self.num_segments, self.num_glance_segments,
                                                  self.num_focus_segments, True, self.device)
            local_images = images[focus_indices].repeat(2, 1, 1, 1)
            local_actions = torch.cat([actions_1[focus_indices], actions_2[focus_indices]], dim=0)
            patches = get_patch_grid_scalexy(local_images, local_actions.detach(), self.patch_size, self.input_size,
                                             self.patch_size)
            with autocast(enabled=self.training):
                local_feat = self.local_CNN.get_featvec(patches)
            local_feat = local_feat.float()
            local_logits = self.local_CNN_fc(local_feat)
            local_logits = local_logits.view(2, -1, self.num_classes)
            local_logits_1 = local_logits[0]
            local_logits_2 = local_logits[1]
            local_feat_sum = local_feat.view(2, -1, self.num_focus_segments, self.local_feature_dim)
            global_feat_sum = global_feat.view(-1, self.num_glance_segments, self.global_feature_dim)
            cat_logits_1, cat_pred_1 = self.classifier(global_feat_sum, local_feat_sum[0])
            cat_logits_2, cat_pred_2 = self.classifier(global_feat_sum, local_feat_sum[1])
            res.append((cat_logits_1, cat_pred_1, global_logits, local_logits_1, temporal_sample_logits, spatial_sample_logits_3, action_3, weights,
                        glance_indices, focus_indices, actions_1[focus_indices]))
            res.append((cat_logits_2, cat_pred_2, global_logits, local_logits_2, temporal_sample_logits, spatial_sample_logits_3, action_3, weights,
                        glance_indices, focus_indices, actions_2[focus_indices]))
            return res
        else:
            focus_indices = policy_sample_indices(weights_T.detach(), self.num_segments, self.num_glance_segments,
                                                  self.num_focus_segments, False, self.device)
            local_images = images[focus_indices]
            local_actions = actions_1[focus_indices]
            patches = get_patch_grid_scalexy(local_images, local_actions.detach(), self.patch_size, self.input_size,
                                             self.patch_size)
            local_feat = self.local_CNN.get_featvec(patches)
            local_logits = self.local_CNN_fc(local_feat)
            local_feat_sum = local_feat.view(-1, self.num_focus_segments, self.local_feature_dim)
            global_feat_sum = global_feat.view(-1, self.num_glance_segments, self.global_feature_dim)
            cat_logits, cat_pred = self.classifier(global_feat_sum, local_feat_sum)
            return cat_logits, cat_pred, global_logits, local_logits, temporal_sample_logits, spatial_sample_logits_3, weights

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    @property
    def crop_size(self):
        return self.input_size

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose(
                [GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])

    def get_optim_policies(self, args):
        return [{'params': self.temporal_policy.parameters(), 'lr_mult': args.temporal_lr_ratio, 'decay_mult': 1,
                 'name': "temporal_policy"}] \
               + [{'params': self.spatial_policy.parameters(), 'lr_mult': args.stn_lr_ratio, 'decay_mult': 1,
                   'name': "spatial_policy"}] \
               + [{'params': self.global_CNN.parameters(), 'lr_mult': args.global_lr_ratio, 'decay_mult': 1,
                   'name': "global_CNN"}] \
               + [{'params': self.global_CNN_fc.parameters(), 'lr_mult': args.global_lr_ratio, 'decay_mult': 1,
                   'name': "global_CNN_fc"}] \
               + [{'params': self.aux_fc.parameters(), 'lr_mult': args.global_lr_ratio, 'decay_mult': 1,
                   'name': "aux_fc"}] \
               + [{'params': self.local_CNN.parameters(), 'lr_mult': 1, 'decay_mult': 1, 'name': "local_CNN"}] \
               + [{'params': self.local_CNN_fc.parameters(), 'lr_mult': 1, 'decay_mult': 1, 'name': "local_CNN_fc"}] \
               + [{'params': self.classifier.parameters(), 'lr_mult': args.classifier_lr_ratio, 'decay_mult': 1,
                   'name': "pooling_classifier"}]


class TemporalPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_glance_segments, temperature, device):
        super().__init__()
        self.device = device
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


class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1)), dim=1)
        return x.max(dim=1)[0]


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, num_neurons=4096):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = [num_neurons]
        layers = []
        dim_input = input_dim
        for dim_output in self.num_neurons:
            layers.append(nn.Linear(dim_input, dim_output))
            layers.append(nn.BatchNorm1d(dim_output))
            layers.append(nn.ReLU())
            dim_input = dim_output
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class PoolingClassifier(nn.Module):
    def __init__(self, global_feature_dim, local_feature_dim, num_glance_segments, num_focus_segments, num_classes,
                 dropout, device):
        super().__init__()
        self.global_feature_dim = global_feature_dim
        self.local_feature_dim = local_feature_dim
        self.num_glance_segments = num_glance_segments
        self.num_focus_segments = num_focus_segments
        self.num_classes = num_classes
        self.max_pooling = MaxPooling()
        self.global_mlp = MultiLayerPerceptron(global_feature_dim)
        self.local_mlp = MultiLayerPerceptron(local_feature_dim)
        self.global_classifiers = nn.ModuleList()
        self.local_classifiers = nn.ModuleList()
        self.device = device
        for m in range(self.num_glance_segments):
            self.global_classifiers.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(4096, self.num_classes)
            ))
        for m in range(self.num_focus_segments):
            self.local_classifiers.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(8192, self.num_classes)
            ))

    def forward(self, x, y):
        _b = x.size(0)
        x = x.view(-1, self.global_feature_dim)
        y = y.view(-1, self.local_feature_dim)
        z1 = self.global_mlp(x).view(_b, self.num_glance_segments, -1)
        z2 = self.local_mlp(y).view(_b, self.num_focus_segments, -1)
        logits = torch.zeros((_b, self.num_glance_segments + self.num_focus_segments, self.num_classes), device=self.device)
        cur_z_1 = z1[:, 0]
        for step_idx in range(0, self.num_glance_segments):
            if step_idx > 0:
                cur_z_1 = self.max_pooling(z1[:, step_idx], cur_z_1)
            logits[:, step_idx] = self.global_classifiers[step_idx](cur_z_1)
        cur_z_2 = z2[:, 0]
        for step_idx in range(0, self.num_focus_segments):
            if step_idx > 0:
                cur_z_2 = self.max_pooling(z2[:, step_idx], cur_z_2)
            logits[:, self.num_glance_segments + step_idx] = self.local_classifiers[step_idx](torch.cat([cur_z_2, cur_z_1], dim=1))
        last_out = logits[:, -1, :].reshape(_b, -1)
        logits = logits.view(_b * (self.num_glance_segments + self.num_focus_segments), -1)
        return logits, last_out


class SpatialPolicy(nn.Module):
    def __init__(self, stn_feature_dim, hidden_dim, num_glance_segments):
        super(SpatialPolicy, self).__init__()
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
                kernel_size=(1, 7, 7), stride=1, padding=(0, 0, 0), bias=False,
            ),
            nn.Sigmoid()
        )

    def forward(self, features):
        NT, C, H, W = features.shape
        features = features.reshape(
            NT // self.num_glance_segments, self.num_glance_segments, C, H, W
        ).permute(0, 2, 1, 3, 4).contiguous()
        actions = self.encoder(features).flatten(2).permute(0, 2, 1).contiguous()
        return actions.reshape(NT, 4)
