# Uni-AdaFocus-X3D with Experiments on Kinetics-400

## Requirements

- python 3.9
- pytorch 1.12.1
- torchvision 0.13.1

## Datasets
1. Please download [Kinetics 400](https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz).

2. Please follow the instructions of [TSM](https://github.com/mit-han-lab/temporal-shift-module#data-preparation) to prepare the Kinetics-400 dataset.

3. After preparation, please edit the "ROOT_DATASET" in `ops/dataset_config.py` to the correct path of the dataset.

## Training

Firstly, download X3D pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1R5lfusmhY7OfqyM7bWw--x7nDeRB8zBE?usp=drive_link) and put them in the `archs/X3D_ckpt` folder.
Run the following command to train Uni-AdaFocus-X3D with patch_size = 128, glance_segments = 16, focus_segments = 32:

### Kinetics-400

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py kinetics RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 16 --num_input_focus_segments 96 --num_focus_segments 32 \
     --gd 20 -j 24 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.005 --wd 5e-5 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --glance_size 160 --patch_size 128 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.40 --temporal_lr_ratio 0.40 --temperature 1.0  --norm_ratio 0.50 
```

## Evaluating Pre-trained Models

* Add "--evaluate" to the training script for the evaluation mode (without sample-wise dynamic).
* Add "--resume PATH_TO_CKPT" to the training script to specify the pre-trained models.
* Add "--test_crops n" to specify the number of crops used for testing.
* After training following the above pipeline, you can get the ckpt `kinetics400_p128_16and32.pth.tar`. Evaluation results of `test_crops=1,3` are as follows.

|File name                          |Dataset        |Patch size |Glance and focus frames    |Acc1   |GFLOPs |
|---                                |---            |---        |---                        |---    |---    |
|kinetics400_p128_16and32.pth.tar   |Kinetics-400   |128        |16 + 32                    |75.3   |8.8*1  |
|kinetics400_p128_16and32.pth.tar   |Kinetics-400   |128        |16 + 32                    |76.6   |8.8*3  |

E.g., Run the following command to evaluate Uni-AdaFocus-X3D with test_crops = 3:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_nViewEval.py kinetics RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 16 --num_input_focus_segments 96 --num_focus_segments 32 \
     --gd 20 -j 24 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.005 --wd 5e-5 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --glance_size 160 --patch_size 128 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.40 --temporal_lr_ratio 0.40 --temperature 1.0  --norm_ratio 0.50 --evaluate --test_crops 3 \
     --resume ckpt/kinetics400_p128_16and32.pth.tar
```

## Evaluating with Sample-wise Dynamic (Early Exit)

To evaluate Uni-AdaFocus with sample-wise dynamic, run the script `test_early_exit_unifocus_X3D_nViewEval.py` with the same evaluation mode
arguments on 1 gpu. E.g.

```
CUDA_VISIBLE_DEVICES=0 python test_early_exit_unifocus_X3D_nViewEval.py kinetics RGB \
     --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
     --root_log LOG_DIR \
     --root_model LOG_DIR \
     --arch resnet50 --num_glance_segments 16 --num_input_focus_segments 96 --num_focus_segments 32 \
     --gd 20 -j 24 --lr_steps 0 0 --lr_type cos --epochs 50 \
     --batch-size 64 --lr 0.005 --wd 5e-5 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
     --glance_size 160 --patch_size 128 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
     --global_lr_ratio 0.5 --stn_lr_ratio 0.40 --temporal_lr_ratio 0.40 --temperature 1.0  --norm_ratio 0.50 --evaluate --test_crops 3 \
     --resume ckpt/kinetics400_p128_16and32.pth.tar
```

## Acknowledgement

- We use the implementation of X3D from [X3D](https://github.com/facebookresearch/SlowFast/tree/main/projects/x3d)
- Feature aggregation modules are borrowed from [FrameExit](https://github.com/Qualcomm-AI-research/FrameExit).
- We borrow some codes for dataset preparation from [AR-Net](https://github.com/mengyuest/AR-Net#dataset-preparation)
