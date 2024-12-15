import torch
import math
from ops.utils import accuracy
import torch.nn.functional as F
import numpy as np


def dynamic_eval_find_threshold(logits, p, vals, sorted_idx):
    """
        logits: m * n * c
        m: Stages
        n: Samples
        c: Classes

        p[k]: ratio of exit at classifier k
        flops[k]: flops of classifier k
        sorted_idx, max_preds, argmax_preds
    """
    p /= p.sum()
    n_stage, n_sample, c = logits.size()

    filtered = torch.zeros(n_sample)
    T = torch.Tensor(n_stage).fill_(1e8)

    for k in range(n_stage - 1):
        acc, count = 0.0, 0
        out_n = math.floor(n_sample * p[k])
        if out_n > 0:
            tmp_sorted_idx = filtered[sorted_idx[k]]
            remain_sorted_idx = sorted_idx[k][tmp_sorted_idx == 0]
            ori_idx = remain_sorted_idx[out_n - 1]
            T[k] = vals[k][ori_idx]
            filtered[remain_sorted_idx[0: out_n]] = 1
    T[n_stage - 1] = -1e8  # accept all of the samples at the last stage

    return T


def evaluate_rand(logits, exp):
    n_stage, n_sample, c = logits.size()
    indices = np.random.permutation(n_sample)
    reverse_indices = torch.tensor([indices.tolist().index(i) for i in range(n_sample)])
    indices = torch.from_numpy(indices)
    logits = logits[:, indices]
    outputs = torch.zeros(n_sample, c)
    cur_pos = 0
    for k in range(n_stage):
        if exp[k] == 0:
            continue
        outputs[cur_pos:cur_pos + int(exp[k])] = logits[k, cur_pos:cur_pos + int(exp[k])]
        cur_pos += int(exp[k])
    outputs = outputs[reverse_indices]
    return outputs


def dynamic_evaluate(logits, flops, T, vals):
    n_stage, n_sample, c = logits.size()
    outputs = torch.zeros(n_sample, c)
    exp = torch.zeros(n_stage)
    acc, expected_flops = 0, 0
    for i in range(n_sample):
        for k in range(n_stage):
            if vals[k][i].item() >= T[k]:  # force the sample to exit at k
                outputs[i] = logits[k, i]
                exp[k] += 1
                break
            if k == n_stage - 1:
                outputs[i] = logits[k]
    for k in range(n_stage):
        _t = 1.0 * exp[k] / n_sample
        expected_flops += _t * flops[k]

    return outputs, expected_flops.item(), exp, 1.0 * exp[0] / n_sample


def confidence(x):
    c, _ = torch.max(x, dim=-1)
    return c


def entropy(x):
    x = F.softmax(x, dim=-1)
    return (x * torch.log(x)).sum(dim=-1)


CRITERIONS = {
    'confidence': confidence,
    'entropy': entropy,
}

### Uni-focus-X3D
X3D_S_16_160_160_gflops = 2.415453044
X3D_L_32_128_128_gflops = 5.984521316
spatial_policy_gflops = 0.376217984
temporal_policy_gflops = 0.001524864

# No classifier
def show_flops(patch_size=128):
    print(f'\nGFLOPs of Unifocus-TSM(early exit), glance frames = 16')
    print(f'patch_size = 128')
    print('=' * 97)
    print(f'|num_steps\t|total GFLOPs\t|global_CNN\t|local_CNN\t|temoral_policy\t|spatial_policy\t|')
    for k in range(0, 33):
        if k != 0 and k != 32:
            continue
        global_gflops = X3D_S_16_160_160_gflops
        local_gflops = 0.0
        temporal_gflops = 0.0
        spatial_gflops = 0.0
        if k > 0:
            local_gflops = X3D_L_32_128_128_gflops
            temporal_gflops = temporal_policy_gflops
            spatial_gflops = spatial_policy_gflops
        Gflops = global_gflops + local_gflops + temporal_gflops + spatial_gflops
        print(f'|{k}\t\t|{Gflops:.9f}\t|{global_gflops:.9f}\t|{local_gflops:.9f}\t|{temporal_gflops:.9f}\t|'
              f'{spatial_gflops:.9f}\t|')
    print('=' * 97)

# TODO

def get_flops(args):
    ps = args.patch_size
    flops = []
    for k in range(0, 2):
        global_gflops = X3D_S_16_160_160_gflops
        local_gflops = 0.0
        temporal_gflops = 0.0
        spatial_gflops = 0.0
        if k > 0:
            local_gflops = X3D_L_32_128_128_gflops
            temporal_gflops = temporal_policy_gflops
            spatial_gflops = spatial_policy_gflops
        Gflops = global_gflops + local_gflops + temporal_gflops + spatial_gflops
        flops.append(Gflops)
    return flops


def early_exit(data, args, criterion_name='confidence'):
    criterion = CRITERIONS[criterion_name]
    flops_exits = get_flops(args)  # List[17]: float
    flops_tot = flops_exits[-1]

    num_exit = 2
    # flops_exits = torch.arange(1, num_exit + 1) * flops_tot / num_exit
    train_logits, train_targets, val_logits, val_targets = data['train_logits'], data['train_targets'], data[
        'val_logits'], data['val_targets']

    # val_targets = val_targets[:, 0]
    last_acc, _ = accuracy(val_logits[:, -1], val_targets, topk=(1, 5))
    print('Flops Tot:', flops_tot)
    print('Acc1:', last_acc.item())

    train_logits = train_logits.permute(1, 0, 2)
    val_logits = val_logits.permute(1, 0, 2)

    train_vals = []
    train_sorted_idx = []
    for k in range(num_exit):
        train_vals.append(criterion(train_logits[k]))
        _, indices = torch.sort(train_vals[k], descending=True, dim=-1)
        train_sorted_idx.append(indices)
    test_vals = []
    for k in range(num_exit):
        test_vals.append(criterion(val_logits[k]))

    flops_list = []
    metric_list = []
    ratio_list = []
    for p in list(range(1, 400)) + [4e3, 4e4, 4e5, 4e6]:
        print(f'distribute p = {p}')
        _p = torch.FloatTensor(1).fill_(p * 1.0 / 100)
        probs = torch.exp(torch.log(_p) * torch.range(0, num_exit - 1))
        probs /= probs.sum()
        T = dynamic_eval_find_threshold(train_logits, probs, train_vals, train_sorted_idx)
        outputs, expected_flops, exp, ratio = dynamic_evaluate(val_logits, flops_exits, T, test_vals)
        # if args.dataset == 'minik':
        acc1, acc5 = accuracy(outputs, val_targets, topk=(1, 5))
        metric_list.append(acc1.item())
        # else:
        #     mAP, _ = cal_map(outputs, val_targets)
        #     metric_list.append(mAP.item())
        flops_list.append(expected_flops)
        ratio_list.append(ratio)

    print('===EARLY EXIT===')
    for f, m, r in zip(flops_list, metric_list, ratio_list):
        print('{:.3f}\t{:.3f}\t{:.3f}'.format(f, m, r))
    return flops_list, metric_list, ratio_list