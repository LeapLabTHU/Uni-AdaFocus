import os
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from ops import dataset_config
from ops.dataset import TSNDataSet
from ops.transforms import *
from ops.early_exit_uni_adafocus import early_exit, show_flops
from opts import parser
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
        ['AdaFocusV2', args.dataset, args.glance_arch, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])
    # if args.rank == 0:
    #     check_rootfolders(args)


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
    # criterion = nn.CrossEntropyLoss().cuda()
    policies = model.module.get_optim_policies(args)
    # specify different optimizer to different training stage
    # optimizer = torch.optim.SGD(policies,
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

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
            # optimizer.load_state_dict(checkpoint['optimizer'])
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
    # partial training
    # if args.dataset == 'minik':
    #     # sample 20k in training set
    #     train_set_index = torch.randperm(len(train_dataset))
    #     sampler = torch.utils.data.sampler.SubsetRandomSampler(train_set_index[-20000:])
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
    #                                                num_workers=args.workers, pin_memory=False, drop_last=True, sampler=sampler)
    # else:
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False,
    #                                                num_workers=args.workers, pin_memory=False, drop_last=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False,
                                               num_workers=args.workers, pin_memory=False, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size * 2, sampler=val_sampler, shuffle=False,
                                             num_workers=args.workers, pin_memory=False)

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
    if args.rank == 0:
        for criterion in ['confidence', 'entropy']:
            print(f'Calculating early exit, criterion = {criterion}...')
            flops_list, metric_list = early_exit(data, args, criterion)
            print(f'Saving early exit...')
            logging = open(os.path.join(args.root_log, args.store_name, f'log_early_exit_{criterion}.csv'), 'a')
            output = 'flops_list,metric_list\n'
            for f, m in zip(flops_list, metric_list):
                output = output + '{:.6f},{:.6f}\n'.format(f, m)
            logging.write(output)
            logging.flush()
            print(f'Flops data saved!')


def cat_and_cpu_and_gather(tensor, args):
    tensor = torch.cat(tensor, 0)
    new_tensor = [torch.zeros_like(tensor),] * dist.get_world_size()
    new_tensor[args.rank] = tensor
    new_tensor = torch.cat(new_tensor, 0)

    new_tensor = new_tensor.clone()
    dist.all_reduce(new_tensor, op=dist.ReduceOp.SUM)

    return new_tensor.cpu()

def generate_logits(val_loader, model, args):
    # switch to evaluate mode
    model.eval()

    all_targets = []
    all_logits = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.rank == 0:
                print(f'[{i}] / [{len(val_loader)}], running')
            _b = target.shape[0]
            target = target.cuda()
            all_targets.append(target)
            images = images.cuda()

            p1 = model(images)
            cat_logits, cat_pred, global_logits, local_logits, _, _, _ = p1
            logits = cat_logits.view(_b, -1, args.num_classes)[:, -17:]
            all_logits.append(logits)

        all_logits = cat_and_cpu_and_gather(all_logits, args)
        all_targets = cat_and_cpu_and_gather(all_targets, args)

    return all_logits, all_targets


if __name__ == '__main__':
    show_flops(128)
    main()
