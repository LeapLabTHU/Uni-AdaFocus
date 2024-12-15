import os
import shutil
import time

import torch.optim
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map
from ops.dataset import TSNDataSet
from ops.transforms import *
from models.uni_adafocus import AdaFocus


def main():
    global args
    args = parser.parse_args()

    ngpus_per_node = 4

    print(ngpus_per_node)

    main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    global best_acc1
    args.batch_size = int(args.batch_size / ngpus_per_node)
    print("Use GPU: {} for training".format(args.local_rank))
    args.rank = args.local_rank

    dist.init_process_group('nccl')

    args.root_model = args.root_log
    best_acc1 = 0.
    if args.glance_arch == 'res50' and args.glance_ckpt_path == '' and not args.evaluate:
        print('WARNING: res50 initialization checkpoint not specified')
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.data_dir)
    args.num_classes = num_class
    args.store_name = '_'.join(
        ['UniAdaFocus', args.dataset, args.glance_arch, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])
    if args.rank == 0:
        check_rootfolders(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AdaFocus(args)
    start_epoch = 0

    scale_size = model.scale_size
    crop_size = model.crop_size
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation(flip=True)
    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    criterion = nn.CrossEntropyLoss().cuda()
    policies = model.module.get_optim_policies(args)
    # specify different optimizer to different training stage
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for group in policies:
        try:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
        except:
            continue

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    cudnn.benchmark = True

    # data loading
    normalize = GroupNormalize(input_mean, input_std)
    train_dataset = TSNDataSet(
        root_path=args.root_path, list_file=args.train_list, num_segments=args.num_segments, image_tmpl=prefix,
        transform=torchvision.transforms.Compose([
            train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize]),
        dense_sample=False,
        dataset=args.dataset,
        partial_fcvid_eval=False,
        partial_ratio=0.2,
        ada_reso_skip=False,
        reso_list=224,
        random_crop=False,
        center_crop=False,
        rescale_to=224,
        policy_input_offset=0,
        save_meta=False)

    val_dataset = TSNDataSet(
        root_path=args.root_path, list_file=args.val_list, num_segments=args.num_segments, image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize]),
        dense_sample=False,
        dataset=args.dataset,
        partial_fcvid_eval=False,
        partial_ratio=0.2,
        ada_reso_skip=False,
        reso_list=224,
        random_crop=False,
        center_crop=False,
        rescale_to=224,
        policy_input_offset=0,
        save_meta=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True, persistent_workers=False)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 1, sampler=val_sampler,
        num_workers=args.workers, pin_memory=False, persistent_workers=False)

    if args.evaluate:
        log_validating = open(os.path.join(args.root_log, args.store_name, f'log_{args.eval_suffix}.csv'), 'a')
        validate(val_loader, model, criterion, args, log_validating)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'a')
    if args.rank == 0:
        with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
            f.write(str(args))

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training)

        if epoch > args.epochs - 10 or (epoch % 5 == 0):
            # evaluate the model on validation set
            acc1 = validate(val_loader, model, criterion, args, log_training)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if args.rank == 0:
                if args.dataset == 'minik':
                    output_best = 'Best Acc@1: %.3f\n' % best_acc1
                else:
                    output_best = 'Best mAP@1: %.3f\n' % best_acc1
                print(output_best)
                log_training.write(output_best + '\n')
                log_training.flush()
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best, args)
        if args.rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc1': best_acc1,
            }, False, args)
        torch.cuda.synchronize()


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def cat_and_cpu_and_gather(tensor, args):
    tensor = torch.cat(tensor, 0)
    new_tensor = [torch.zeros_like(tensor),] * dist.get_world_size()
    new_tensor[args.rank] = tensor
    new_tensor = torch.cat(new_tensor, 0)

    new_tensor = new_tensor.clone()
    dist.all_reduce(new_tensor, op=dist.ReduceOp.SUM)

    return new_tensor.cpu()


def validate(val_loader, model, criterion, args, log=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()

    all_result = []
    all_soft_result = []
    all_targets = []
    all_local_result = []
    all_global_result = []
    all_cat_result = []
    end = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            _b = target.shape[0]
            images = images.cuda()
            target = target.cuda()
            all_targets.append(target)
            target = target[:, 0]

            target_c = target.view(_b, -1).expand(_b, args.num_glance_segments + args.num_focus_segments).reshape(-1)
            p1 = model(images)
            cat_logits, cat_pred, global_logits, local_logits, _, _, w = p1

            loss = criterion(cat_logits, target_c)
            all_result.append(cat_pred)
            soft = nn.Softmax(dim=1)
            all_soft_result.append(soft(cat_pred))

            local_logits = local_logits.view(-1, args.num_focus_segments, args.num_classes)
            global_logits = global_logits.view(-1, args.num_glance_segments, args.num_classes)
            cat_logits = cat_logits.view(-1, args.num_glance_segments + args.num_focus_segments, args.num_classes)
            soft = nn.Softmax(dim=2)
            local_logits = soft(local_logits).mean(dim=1)
            global_logits = soft(global_logits).mean(dim=1)
            cat_logits = soft(cat_logits).mean(dim=1)

            all_local_result.append(local_logits)
            all_global_result.append(global_logits)
            all_cat_result.append(cat_logits)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(cat_pred, target, topk=(1, 5))
            loss = reduce_tensor(loss)
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.rank == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

        all_result = cat_and_cpu_and_gather(all_result, args)
        all_soft_result = cat_and_cpu_and_gather(all_soft_result, args)
        all_local_result = cat_and_cpu_and_gather(all_local_result, args)
        all_global_result = cat_and_cpu_and_gather(all_global_result, args)
        all_cat_result = cat_and_cpu_and_gather(all_cat_result, args)
        all_targets = cat_and_cpu_and_gather(all_targets, args)

        if args.rank == 0:
            mAP, AP_list = cal_map(all_result, all_targets)
            smAP, _ = cal_map(all_soft_result, all_targets)
            local_mAP, _ = cal_map(all_local_result, all_targets)
            global_mAP, _ = cal_map(all_global_result, all_targets)
            cat_mAP, _ = cal_map(all_cat_result, all_targets)
            output = ('(rank={rank}) Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}\n'
                      'mAP: {mAP} smAP: {smAP} Local_mAP: {Local_mAP} Global_mAP: {Global_mAP} '
                      'Cat_mAP: {Cat_mAP}'
                      .format(rank=args.rank, top1=top1, top5=top5, loss=losses, mAP=mAP, smAP=smAP, Local_mAP=local_mAP,
                              Global_mAP=global_mAP, Cat_mAP=cat_mAP))
            for i in range(10):
                output = output + f' ,AP[{i}]={AP_list[i]}'
            print(output)
            if log is not None:
                log.write(output + '\n')
                log.flush()
            if args.dataset == 'minik':
                return top1.avg
            else:
                return mAP
        return 0


def train(train_loader, model, criterion, optimizer, epoch, args, log=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_g = AverageMeter('Global Loss', ':.4e')
    losses_l = AverageMeter('Local Loss', ':.4e')
    losses_t = AverageMeter('Temporal Loss', ':.4e')
    losses_s = AverageMeter('Spatial Loss', ':.4e')
    losses_KL = AverageMeter('KL Loss', ':.4e')
    losses_norm = AverageMeter('Norm Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()

    end = time.time()

    all_targets = []

    for i, (images, target) in enumerate(train_loader):
        _b = target.shape[0]
        all_targets.append(target)
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target[:, 0].cuda()

        optimizer.zero_grad()
        p1, p2 = model(images)
        outputs_1, pred_1, outputs_global_1, outputs_local_1, outputs_temporal_1, outputs_spatial_1, action3, w1, gind1, lind1, act1 = p1  # [BT, C], [B, C], [BT, C], [BT, C]
        outputs_2, pred_2, outputs_global_2, outputs_local_2, outputs_temporal_2, outputs_spatial_2, _______, w2, gind2, lind2, act2 = p2
        target_g = target.view(_b, -1).expand(_b, args.num_glance_segments).reshape(-1)
        target_l = target.view(_b, -1).expand(_b, args.num_focus_segments).reshape(-1)
        target_c = target.view(_b, -1).expand(_b, args.num_glance_segments + args.num_focus_segments).reshape(-1)

        loss_cat = criterion(outputs_1, target_c) + criterion(outputs_2, target_c)
        loss_global = criterion(outputs_global_1, target_g) + criterion(outputs_global_2, target_g)
        loss_local = criterion(outputs_local_1, target_l) + criterion(outputs_local_2, target_l)
        loss_temporal = criterion(outputs_temporal_1, target) + criterion(outputs_temporal_2, target)
        loss_spatial = criterion(outputs_spatial_1, target) + criterion(outputs_spatial_2, target)

        uniform_dist = torch.ones_like(w1) / args.num_focus_segments
        KL_div = torch.nn.functional.kl_div(uniform_dist.log(), w1, reduction='batchmean')
        loss_KL = KL_div * args.KL_ratio

        act_target = torch.ones_like(action3[:, 2:4]) * 1.0
        loss_norm = ((action3[:, 2:4] - act_target) ** 2).mean()
        loss = (loss_cat + loss_global + loss_local + loss_temporal + loss_spatial) / 2 + loss_KL + loss_norm * args.norm_ratio

        loss.backward()
        optimizer.step()

        # Update evaluation metrics
        acc1, acc5 = accuracy(pred_1, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        losses_g.update(loss_global.item(), images.size(0))
        losses_l.update(loss_local.item(), images.size(0))
        losses_t.update(loss_temporal.item(), images.size(0))
        losses_s.update(loss_spatial.item(), images.size(0))
        losses_KL.update(KL_div.item(), images.size(0))
        losses_norm.update(loss_norm.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Global Loss {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
                      'Local Loss {loss_l.val:.4f} ({loss_l.avg:.4f})\t'
                      'Temporal Loss {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                      'Spatial Loss {loss_s.val:.4f} ({loss_s.avg:.4f})\t'
                      'KL Loss {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t'
                      'Norm Loss {loss_norm.val:.4f} ({loss_norm.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                loss=losses, loss_g=losses_g, loss_l=losses_l, loss_t=losses_t, loss_s=losses_s, loss_KL=losses_KL,
                loss_norm=losses_norm,
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders(args):
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


def save_checkpoint(state, is_best, args):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


if __name__ == '__main__':
    main()
