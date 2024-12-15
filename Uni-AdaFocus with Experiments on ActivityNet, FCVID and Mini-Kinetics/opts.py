import argparse

parser = argparse.ArgumentParser('Adafocus V2')

parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--root_log', type=str, default='exp')
parser.add_argument('--dataset', type=str, default='actnet', help='actnet, fcvid, minik')

parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--num_segments', type=int, default=48)
parser.add_argument('--num_glance_segments', type=int, default=16)
parser.add_argument('--num_focus_segments', type=int, default=16)
parser.add_argument('--num_steps', type=int, default=1, help='num of steps, of mus')
parser.add_argument('--sample_sigma', type=float, default=0.015, help='sample ~ N(mu, sigma)')

parser.add_argument('--glance_arch', type=str, default='mbv2', help='mbv2, res50')
parser.add_argument('--glance_ckpt_path', type=str, default='', help='for glancer res50')

parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--fc_dropout', type=float, default=0.2)

parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', type=int, default=1007)

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--lr_type', type=str, default='cos')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--global_lr_ratio', default=0.5, type=float)
parser.add_argument('--stn_lr_ratio', default=0.2, type=float)
parser.add_argument('--temporal_lr_ratio', default=0.2, type=float)
parser.add_argument('--classifier_lr_ratio', default=20.0, type=float)
parser.add_argument('--KL_ratio', default=0.0, type=float)

parser.add_argument('--temperature', default=1.0, type=float)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=2e-4)

parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--patch_size', type=int, default=96)
parser.add_argument('--glance_size', type=int, default=96)
parser.add_argument('--hidden_dim', type=int, default=1024, help='for Classifer')
parser.add_argument('--stn_hidden_dim', type=int, default=128, help='for STN')
parser.add_argument('--temporal_hidden_dim', type=int, default=64, help='for TemporalPolicy')

parser.add_argument('--sample', action='store_true', default=False)
parser.add_argument('--save', type=str, default='')

parser.add_argument('--local_rank', type=int, default=0)

# Temporal Configuration
parser.add_argument('--multi_sample_times', type=int, default=3, help='for multinomial sample')
parser.add_argument('--eval_suffix', type=str, default='eval')

# Spatial Configuration
parser.add_argument('--scale_factor', default=0.3, type=float)
parser.add_argument('--norm_ratio', default=1.0, type=float)
parser.add_argument('--normal_sig', default=0.0, type=float)

parser.add_argument('--fold_id', default=None, type=int, help='for N-fold testing')