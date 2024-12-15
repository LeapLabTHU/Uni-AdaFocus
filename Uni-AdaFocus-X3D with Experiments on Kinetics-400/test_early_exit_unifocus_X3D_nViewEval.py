import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.optim
from torch.cuda.amp import GradScaler

from ops import dataset_config
from opts import parser
from ops.dataset import TSNDataSet
from ops.transforms import *
from archs.uni_adafocus_X3D_exit import AdaFocus
from ops.early_exit_unifocus_X3D import early_exit, show_flops


def main():
    global args
    args = parser.parse_args()

    # create model
    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(
        args.dataset,
        args.modality)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaFocus(num_class, args)
    input_size = model.local_CNN.input_size
    crop_size = model.local_CNN.crop_size
    scale_size = model.local_CNN.scale_size
    input_mean = model.local_CNN.input_mean
    input_std = model.local_CNN.input_std
    policies = model.get_optim_policies(args)
    train_augmentation = model.local_CNN.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    args.distributed = None
    ngpus_per_node = 1
    args.rank = 0

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

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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
                print(("=> evaluating loading checkpoint '{}'".format(args.resume)))
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

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, scale_size, flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, scale_size, flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, scale_size)
        ])
    else:
        raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))

    val_dataset = TSNDataSet(
        args.root_path, args.val_list,
        num_segments_glancer=args.num_glance_segments,
        num_segments_focuser=args.num_input_focus_segments,
        new_length=data_length,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            cropping,
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
    print(f'train_sampler={train_sampler}')
    print(f'val_sampler={val_sampler}')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler,
        persistent_workers=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler,
        persistent_workers=False)

    print('Generate Logits on Valset')
    val_logits, val_targets = generate_logits(val_loader, model, args)
    print('Generate Logits on Trainset')
    train_logits, train_targets = generate_logits(train_loader, model, args)
    data = {
        'train_logits': train_logits.cpu(),
        'train_targets': train_targets.cpu(),
        'val_logits': val_logits.cpu(),
        'val_targets': val_targets.cpu()
    }

    for criterion in ['confidence', 'entropy']:
        print(f'Calculating early exit, criterion = {criterion}...')
        flops_list, metric_list, ratio_list = early_exit(data, args, criterion)
        print(f'Saving early exit...')
        logging = open(os.path.join(args.root_log, args.store_name, f'log_early_exit_{args.test_crops}views_{criterion}.csv'), 'a')
        output = 'flops_list,metric_list\n'
        for f, m, r in zip(flops_list, metric_list, ratio_list):
            output = output + '{:.6f},{:.6f},{:.6f}\n'.format(f, m, r)
        logging.write(output)
        logging.flush()
        print(f'Flops data saved!')


def generate_logits(val_loader, model, args):
    # switch to evaluate mode
    model.eval()
    print(f'Start generating, args.rank={args.rank}')
    all_targets = []
    all_logits = []
    with torch.no_grad():
        for i, (input_glance, images_input, target) in enumerate(val_loader):
            print(f'[{i}] / [{len(val_loader)}], running')
            _b = target.shape[0]
            all_targets.append(target)
            input_glance = input_glance.cuda(args.gpu, non_blocking=True)
            images_input = images_input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            p1 = model(images_glance=input_glance, images_input=images_input)
            glogit, logit, _, _, _, _, _, _ = p1
            b_glogit = glogit.reshape(-1, args.test_crops, glogit.size(-1)).mean(1)
            b_logit = logit.reshape(-1, args.test_crops, logit.size(-1)).mean(1)
            logits = torch.stack([b_glogit, b_logit], dim=1)
            all_logits.append(logits)
    all_logits = torch.cat(all_logits, 0)
    all_targets = torch.cat(all_targets, 0)
    return all_logits, all_targets


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
    show_flops(128)
    main()
