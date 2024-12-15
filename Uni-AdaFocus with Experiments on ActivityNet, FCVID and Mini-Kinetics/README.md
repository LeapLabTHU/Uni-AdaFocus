# Uni-AdaFocus with Experiments on ActivityNet, FCVID and Mini-Kinetics

## Requirements

- python 3.9
- pytorch 1.12.1
- torchvision 0.13.1

## Datasets
1. Please get the train/test split files for each dataset from [Google Drive](https://drive.google.com/drive/folders/1QZ2gVoGMh3Xe20kgdG6s7cnQTYsDWQXz?usp=sharing) and put them in `PATH_TO_DATASET`.
2. Download videos from following links, or contact the corresponding authors for the access. 

   - [ActivityNet-v1.3](http://activity-net.org/download.html) 
   - [FCVID](https://drive.google.com/drive/folders/1cPSc3neTQwvtSPiVcjVZrj0RvXrKY5xj)
   - [Mini-Kinetics](https://deepmind.com/research/open-source/kinetics). Please download [Kinetics 400](https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz) and select the 200 classes for minik.

3. Extract frames using [ops/video_jpg.py](ops/video_jpg.py). Minor modifications on file path are needed when extracting frames from different datasets. You may also need to change the dataset configs in `dataset_config.py`  as well.

## Training

Run the following command to train Uni-AdaFocus with patch_size = 128, glance_segments = 16, focus_segments = 16:

### ActivityNet

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main.py \
  --glance_arch mbv2 --dataset actnet --data_dir PATH_TO_DATASET   \
  --root_log LOG_DIR  \
  --workers 16 --num_segments 48 --dropout 0.2  --fc_dropout 0.2   \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0   \
  --batch_size 32 --momentum 0.9 --weight_decay 2e-4   \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.2 --classifier_lr_ratio 20.0 --temporal_lr_ratio 0.2 --KL_ratio 0.0 --norm_ratio 1.00  \
  --hidden_dim 1024 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
  --num_glance_segments 16 --num_focus_segments 16 --temperature 1.0 
```

### FCVID

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main_semiAMP.py \
  --glance_arch mbv2 --dataset fcvid --data_dir PATH_TO_DATASET   \
  --root_log LOG_DIR  \
  --workers 16 --num_segments 48 --dropout 0.2  --fc_dropout 0.2   \
  --epochs 50 --lr 0.004 --lr_type cos --lr_steps 0 0   \
  --batch_size 64 --momentum 0.9 --weight_decay 1e-4   \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.2 --classifier_lr_ratio 20.0 --temporal_lr_ratio 0.2 --KL_ratio 0.0 --norm_ratio 1.00  \
  --hidden_dim 1024 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
  --num_glance_segments 16 --num_focus_segments 16 --temperature 1.0 --print-freq 50 
```

### Mini-Kinetics

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main_semiAMP.py \
  --glance_arch mbv2 --dataset minik --data_dir PATH_TO_DATASET \
  --root_log LOG_DIR \
  --workers 16 --num_segments 48 --dropout 0.2  --fc_dropout 0.1 \
  --epochs 50 --lr 0.003 --lr_type cos --lr_steps 0 0 \
  --batch_size 48 --momentum 0.9 --weight_decay 1e-4 \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.2 --classifier_lr_ratio 20.0 --temporal_lr_ratio 0.2 --KL_ratio 0.0 --norm_ratio 0.75  \
  --hidden_dim 1024 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
  --num_glance_segments 16 --num_focus_segments 16 --temperature 1.0 --print-freq 50
```

## Evaluating Pre-trained Models

* Add "--evaluate" to the training script for the evaluation mode (without sample-wise dynamic).
* Add "--resume PATH_TO_CKPT" to the training script to specify the pre-trained models.
* We provide the following pre-trained models. All pre-trained models are available at [Google Drive](https://drive.google.com/drive/folders/1UsUSKjESAhfVGQV-CRPiUXxB16nvANnh?usp=drive_link). Please download the corresponding pre-trained models and put them in the `ckpt` folder.

|File name           |Dataset       |Patch size |mAP or Acc1(for minik) |GFLOPs  |
|---                 |---           |---        |---                    |---     |
|actnet_p128.pth.tar |ActivityNet   |128        |80.7                   |27.2    |
|fcvid_p128.pth.tar  |FCVID         |128        |86.4                   |27.2    |
|minik_p128.pth.tar  |Mini-Kinetics |128        |75.8                   |27.2    |


## Evaluating with Sample-wise Dynamic (Early Exit)

To evaluate Uni-AdaFocus with sample-wise dynamic, run the script `test_early_exit_uni_adafocus.py` with the same evaluation mode
arguments. E.g.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 test_early_exit_uni_adafocus.py \
  --glance_arch mbv2 --dataset actnet --data_dir PATH_TO_DATASET   \
  --root_log LOG_DIR  \
  --workers 16 --num_segments 48 --dropout 0.2  --fc_dropout 0.2   \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0   \
  --batch_size 32 --momentum 0.9 --weight_decay 2e-4   \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.2 --classifier_lr_ratio 20.0 --temporal_lr_ratio 0.2 --KL_ratio 0.0 --norm_ratio 1.00  \
  --hidden_dim 1024 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
  --num_glance_segments 16 --num_focus_segments 16 --temperature 1.0 --evaluate \
  --resume ckpt/actnet_p128.pth.tar
```

## Acknowledgement

- We use the implementation of MobileNet-v2 and ResNet from [Pytorch source code](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html). 
- Feature aggregation modules are borrowed from [FrameExit](https://github.com/Qualcomm-AI-research/FrameExit).
- We borrow some codes for dataset preparation from [AR-Net](https://github.com/mengyuest/AR-Net#dataset-preparation)
