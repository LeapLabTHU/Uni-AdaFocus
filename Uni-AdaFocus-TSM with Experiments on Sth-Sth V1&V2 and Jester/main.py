import os
import time
import shutil
import torch.optim
import torch.nn.parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from ops import dataset_config
from opts import parser
from ops.utils import AverageMeter, accuracy
from ops.dataset import TSNDataSet
from tensorboardX import SummaryWriter
from ops.transforms import *
from torch.cuda.amp import autocast, GradScaler
from archs.uni_adafocus_tsm import AdaFocus

best_prec1 = 0


def main():
    global args
    args = parser.parse_args()
    cudnn.benchmark = False

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_prec1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(
        args.dataset,
        args.modality)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaFocus(num_class, args)
    crop_size = model.local_CNN.crop_size
    scale_size = model.local_CNN.scale_size
    input_mean = model.local_CNN.input_mean
    input_std = model.local_CNN.input_std
    policies = model.get_optim_policies(args)
    train_augmentation = model.local_CNN.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scaler = GradScaler()

    for group in policies:
        try:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
        except:
            continue

    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_glance_segments,
        'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        print('storing name: ' + args.store_name)
        check_rootfolders(args)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            raise NotImplementedError
            # make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            if args.evaluate:
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print(("=> loading checkpoint '{}'".format(args.resume)))
                checkpoint = torch.load(args.resume, map_location='cpu')
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scaler.load_state_dict(checkpoint['scaler'])
                print(("=> loaded checkpoint '{}' (epoch {})"
                       .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        raise NotImplementedError
        # make_temporal_pool(model.module.base_model, args.num_segments)

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_dataset = TSNDataSet(
        args.root_path, args.train_list,
        num_segments_glancer=args.num_glance_segments,
        num_segments_focuser=args.num_input_focus_segments,
        new_length=data_length,
        modality=args.modality,
        image_tmpl=prefix,
        transform=torchvision.transforms.Compose([
            train_augmentation,
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize,
        ]), dense_sample=args.dense_sample)
    
    val_dataset = TSNDataSet(
        args.root_path, args.val_list,
        num_segments_glancer=args.num_glance_segments,
        num_segments_focuser=args.num_input_focus_segments,
        new_length=data_length,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize,
        ]), dense_sample=args.dense_sample)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
            persistent_workers=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler,
            persistent_workers=False)

    if args.evaluate:
        validate(args, val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'a')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(args, optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(args, train_loader, model, scaler, criterion, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(args, val_loader, model, criterion, epoch, log_training, tf_writer)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

                output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
                print(output_best)
                log_training.write(output_best + '\n')
                log_training.flush()

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                    'scaler': scaler.state_dict(),
                }, is_best, args)


def train(args, train_loader, model, scaler, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_target = AverageMeter()
    losses_global = AverageMeter()
    losses_local = AverageMeter()
    losses_temporal = AverageMeter()
    losses_spatial = AverageMeter()
    losses_norm = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.global_CNN.partialBN(False)
        model.module.local_CNN.partialBN(False)
    else:
        model.module.global_CNN.partialBN(True)
        model.module.local_CNN.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_glance, images_input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_glance = input_glance.cuda(args.gpu, non_blocking=True)
        images_input = images_input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        optimizer.zero_grad()
        with autocast():
            p1, p2 = model(images_glance=input_glance, images_input=images_input)

            logit_1, global_logit_1, local_logit_1, temporal_logit_1, spatial_logit_1, action_1, action_3, w1, lind1 = p1
            logit_2, global_logit_2, local_logit_2, temporal_logit_2, spatial_logit_2, action_2, ________, w2, lind2 = p2

            loss_target = criterion(logit_1, target) + criterion(logit_2, target)
            loss_global = criterion(global_logit_1, target) + criterion(global_logit_2, target)
            loss_local = criterion(local_logit_1, target) + criterion(local_logit_2, target)
            loss_temporal = criterion(temporal_logit_1, target) + criterion(temporal_logit_2, target)
            loss_spatial = criterion(spatial_logit_1, target) + criterion(spatial_logit_2, target)

            act_target = torch.ones_like(action_3[:, 2:4]) * 1.0
            loss_norm = ((action_3[:, 2:4] - act_target) ** 2).mean()

            loss = (loss_target + loss_global + loss_local + loss_temporal + loss_spatial) / 2 + loss_norm * args.norm_ratio

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logit_2.data, target, topk=(1, 5))
        losses.update(loss.item(), input_glance.size(0))
        losses_target.update(loss_target.item(), input_glance.size(0))
        losses_global.update(loss_global.item(), input_glance.size(0))
        losses_local.update(loss_local.item(), input_glance.size(0))
        losses_temporal.update(loss_temporal.item(), input_glance.size(0))
        losses_spatial.update(loss_spatial.item(), input_glance.size(0))
        losses_norm.update(loss_norm.item(), input_glance.size(0))
        top1.update(prec1.item(), input_glance.size(0))
        top5.update(prec5.item(), input_glance.size(0))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank == 0):
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Global Loss {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
                        'Local Loss {loss_l.val:.4f} ({loss_l.avg:.4f})\t'
                        'Temporal Loss {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                        'Spatial Loss {loss_s.val:.4f} ({loss_s.avg:.4f})\t'
                        'Norm Loss {loss_norm.val:.4f} ({loss_norm.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, loss_g=losses_global, loss_l=losses_local, loss_t=losses_temporal, loss_s=losses_spatial, loss_norm=losses_norm,
                    top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
                print(output)
                print(f'weights = {w1[0]} '
                      f'local indices = {lind1[:2*args.num_focus_segments]}, actions1 = {action_1[:2]}, actions2 = {action_2[:2]}, actions3 = {action_3[:2]}')

                log.write(output + '\n')
                log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(args, val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_list = [AverageMeter() for i in range(5)]
    top5_list = [AverageMeter() for i in range(5)]

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input_glance, images_input, target) in enumerate(val_loader):

            input_glance = input_glance.cuda(args.gpu, non_blocking=True)
            images_input = images_input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            logit, global_logit, local_logit, temporal_logit, spatial_logit, action, action_3, w, lind = model(images_glance=input_glance, images_input=images_input)
            logit_list = [logit, global_logit, local_logit, temporal_logit, spatial_logit]
            loss = criterion(logit, target)
            loss = reduce_tensor(loss)
            losses.update(loss.item(), input_glance.size(0))

            # measure accuracy and record loss
            for a_logit, top1, top5 in zip(logit_list, top1_list, top5_list):
                prec1, prec5 = accuracy(a_logit.data, target, topk=(1, 5))
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
                top1.update(prec1.item(), input_glance.size(0))
                top5.update(prec5.item(), input_glance.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank == 0):
                    output = ('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1_list[0], top5=top5_list[0]))
                    print(output)
                    if log is not None:
                        log.write(output + '\n')
                        log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}\t'
              'Global Prec@1 {g_top1.avg:.3f} Global Prec@5 {g_top5.avg:.3f}\t'
              'Local Prec@1 {l_top1.avg:.3f} Local Prec@5 {l_top5.avg:.3f}\t'
              'Temporal Prec@1 {t_top1.avg:.3f} Temporal Prec@5 {t_top5.avg:.3f}\t'
              'Spatial Prec@1 {s_top1.avg:.3f} Spatial Prec@5 {s_top5.avg:.3f}\t'
              .format(top1=top1_list[0], top5=top5_list[0], loss=losses,
                      g_top1=top1_list[1], g_top5=top5_list[1],
                      l_top1=top1_list[2], l_top5=top5_list[2],
                      t_top1=top1_list[3], t_top5=top5_list[3],
                      s_top1=top1_list[4], s_top5=top5_list[4],
                      ))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1_list[0].avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5_list[0].avg, epoch)

    return top1_list[0].avg


def save_checkpoint(state, is_best, args):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(args, optimizer, epoch, lr_type, lr_steps):
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


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def check_rootfolders(args):
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
