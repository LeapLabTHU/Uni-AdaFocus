# Uni-AdaFocus-TSM with Experiments on Something-Something V1&V2 and Jester

## Requirements

- python 3.9
- pytorch 1.12.1
- torchvision 0.13.1

## Datasets
1. Please follow the instructions of [TSM](https://github.com/mit-han-lab/temporal-shift-module#data-preparation) to prepare the Something-Something V1/V2 and Jester datasets.

2. After preparation, please edit the "ROOT_DATASET" in `ops/dataset_config.py` to the correct path of the dataset.

## Training

Run the following command to train Uni-AdaFocus-TSM with varying dataset (i.e., [something/somethingv2/jester]),
patch_size (i.e., [96, 128]), glance_segments and focus_segments (i.e., [8 + 8, 8 + 12, 8 + 16]):

### Something-Something V1, p96, 8 + 8
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py something RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 8 --num_input_focus_segments 24 --num_focus_segments 8 \
     --gd 20 -j 16 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.02 --wd 1e-3 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --patch_size 96 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.20 --temporal_lr_ratio 0.20 --temperature 1.0  --norm_ratio 0.50
```

### Something-Something V1, p96, 8 + 12
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py something RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 8 --num_input_focus_segments 36 --num_focus_segments 12 \
     --gd 20 -j 16 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.02 --wd 1e-3 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --patch_size 96 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.20 --temporal_lr_ratio 0.20 --temperature 1.0  --norm_ratio 0.50 
```

### Something-Something V1, p128, 8 + 12
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py something RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 8 --num_input_focus_segments 36 --num_focus_segments 12 \
     --gd 20 -j 16 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.02 --wd 1e-3 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --patch_size 128 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.20 --temporal_lr_ratio 0.20 --temperature 1.0  --norm_ratio 0.50 
```

### Something-Something V1, p128, 8 + 16
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py something RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 8 --num_input_focus_segments 48 --num_focus_segments 16 \
     --gd 20 -j 16 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.02 --wd 1e-3 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --patch_size 128 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.20 --temporal_lr_ratio 0.20 --temperature 1.0  --norm_ratio 0.50 
```

### Something-Something V2, p96, 8 + 8
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py somethingv2 RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 8 --num_input_focus_segments 24 --num_focus_segments 8 \
     --gd 20 -j 16 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.02 --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --patch_size 96 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.20 --temporal_lr_ratio 0.20 --temperature 1.0  --norm_ratio 0.50 
```

### Something-Something V2, p96, 8 + 12
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py somethingv2 RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 8 --num_input_focus_segments 36 --num_focus_segments 12 \
     --gd 20 -j 16 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.02 --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --patch_size 96 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.20 --temporal_lr_ratio 0.20 --temperature 1.0  --norm_ratio 0.50 
```

### Something-Something V2, p128, 8 + 12
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py somethingv2 RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 8 --num_input_focus_segments 36 --num_focus_segments 12 \
     --gd 20 -j 12 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.02 --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --patch_size 128 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.20 --temporal_lr_ratio 0.20 --temperature 1.0  --norm_ratio 0.50 
```

### Something-Something V2, p128, 8 + 16
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py somethingv2 RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 8 --num_input_focus_segments 48 --num_focus_segments 16 \
     --gd 20 -j 16 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.02 --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --patch_size 128 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.20 --temporal_lr_ratio 0.20 --temperature 1.0  --norm_ratio 0.50
```

### Jester, p128, 8 + 12 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py jester RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 8 --num_input_focus_segments 36 --num_focus_segments 12 \
     --gd 20 -j 16 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.02 --wd 1e-3 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --patch_size 128 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.20 --temporal_lr_ratio 0.20 --temperature 1.0  --norm_ratio 0.25 
```

## Evaluating Pre-trained Models

* Add "--evaluate" to the training script for the evaluation mode (without sample-wise dynamic).
* Add "--resume PATH_TO_CKPT" to the training script to specify the pre-trained models.
* We provide the following pre-trained models. All pre-trained models are available at [Google Drive](https://drive.google.com/drive/folders/1tFbZd2ZYt55_QYxZDHvAwT4nNaKTHCbB?usp=drive_link). Please download the corresponding pre-trained models and put them in the `ckpt` folder.

|File name                  |Dataset        |Patch size |Glance and focus frames    |Acc1    |GFLOPs |
|---                        |---            |---        |---                        |---    |---    |
|sthv1_p96_8and8.pth.tar    |Sth-Sth V1     |96         |8 + 8                      |47.5   |8.8    |
|sthv1_p96_8and12.pth.tar   |Sth-Sth V1     |96         |8 + 12                     |49.5   |11.8   |
|sthv1_p128_8and12.pth.tar  |Sth-Sth V1     |128        |8 + 12                     |50.5   |18.8   |
|sthv1_p128_8and16.pth.tar  |Sth-Sth V1     |128        |8 + 16                     |51.5   |24.2   |
|sthv2_p96_8and8.pth.tar    |Sth-Sth V2     |96         |8 + 8                      |60.8   |8.8    |
|sthv2_p96_8and12.pth.tar   |Sth-Sth V2     |96         |8 + 12                     |62.6   |11.8   |
|sthv2_p128_8and12.pth.tar  |Sth-Sth V2     |128        |8 + 12                     |63.2   |18.8   |
|sthv2_p128_8and16.pth.tar  |Sth-Sth V2     |128        |8 + 16                     |64.2   |24.2   |
|jester_p128_8and12.pth.tar |Jester         |128        |8 + 12                     |97.1   |18.8   |


## Evaluating with Sample-wise Dynamic (Early Exit)

To evaluate Uni-AdaFocus with sample-wise dynamic, run the script `test_early_exit_uni_adafocus_tsm.py` with the same evaluation mode
arguments. E.g.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python test_early_exit_uni_adafocus_tsm.py something RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 8 --num_input_focus_segments 36 --num_focus_segments 12 \
     --gd 20 -j 16 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.02 --wd 1e-3 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --patch_size 128 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.20 --temporal_lr_ratio 0.20 --temperature 1.0  --norm_ratio 0.50 --evaluate \
     --resume ckpt/sthv1_p128_8and12.pth.tar
```

## Acknowledgement

- We use the implementation of MobileNet-v2 and ResNet from [Pytorch source code](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html). 
- Our code is based on the official implementation of [temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module).