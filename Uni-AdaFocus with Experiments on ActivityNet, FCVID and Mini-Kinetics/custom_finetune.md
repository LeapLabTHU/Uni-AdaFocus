# Custom finetune

If you have a customized video classification dataset and would like to finetune Uni-AdaFocus on it, please follow this tutorial step-by-step.

Let's begin with this folder [Uni-AdaFocus with Experiments on ActivityNet, FCVID and Mini-Kinetics](Uni-AdaFocus%20with%20Experiments%20on%20ActivityNet,%20FCVID%20and%20Mini-Kinetics).

## Prepare dataset

1. First prepare your dataset in this structure:
```
PATH_TO_YOUR_DATASET/
│
├── classInd.txt
├── train_split.txt
├── val_split.txt
├── frames/
│   ├── video1/
│   │   ├── image_00001.jpg
│   │   ├── image_00002.jpg
│   │   ├── ...
│   ├── video2/
│   │   ├── image_00001.jpg
│   │   ├── image_00002.jpg
│   │   ├── ...
│   ├── ...
```
2. Format annotations

Make sure the format of "classInd.txt", "train_split.txt", "val_split.txt" be similar to that of [actnet](https://drive.google.com/drive/folders/1bY0Cdrl72PdbC_5aHXtfqYunNzs_45Cq).
* classInd.txt: class name index file, one line for each class.
* train_split.txt: train set labels, `<video_name>,<num_frames>,<class_id>`
* val_split.txt: valid set labels, `<video_name>,<num_frames>,<class_id>`


3. Extract video frames

You can use [ops/video_jpg.py](ops/video_jpg.py) to extract video frames

```
python ops/video_jpg.py PATH_TO_YOUR_DATASET/videos PATH_TO_YOUR_DATASET/frames --parallel
```

## Update dataset config

1. Add your own dataset config in [ops/dataset_config.py](ops/dataset_config.py): (replace YOUR_DATA with your own dataset name)
```
def return_YOUR_DATA(data_dir):
    filename_categories = ospj(data_dir, 'classInd.txt')
    root_data = data_dir + "/frames"
    filename_imglist_train = ospj(data_dir, 'train_split.txt')
    filename_imglist_val = ospj(data_dir, 'val_split.txt')
    prefix = 'image_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
```

2. Change [this line](ops/dataset_config.py#L34) to:
```
dict_single = {'actnet': return_actnet, 'fcvid': return_fcvid, 'minik': return_minik}
↓
dict_single = {'actnet': return_actnet, 'fcvid': return_fcvid, 'minik': return_minik, 'YOUR_DATA': return_YOUR_DATA}
```

## Start training

Run the following command to train Uni-AdaFocus on the train split of your dataset.

* YOUR_DATA: your dataset name, same to above
* PATH_TO_YOUR_DATASET: path to your dataset
* LOG_DIR: path to the directory for saving experimental results

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main.py \
  --glance_arch mbv2 --dataset YOUR_DATA --data_dir PATH_TO_YOUR_DATASET   \
  --root_log LOG_DIR  \
  --workers 16 --num_segments 48 --dropout 0.2  --fc_dropout 0.2   \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0   \
  --batch_size 32 --momentum 0.9 --weight_decay 2e-4   \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.2 --classifier_lr_ratio 20.0 --temporal_lr_ratio 0.2 --KL_ratio 0.0 --norm_ratio 1.00  \
  --hidden_dim 1024 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
  --num_glance_segments 16 --num_focus_segments 16 --temperature 1.0 
```

## Start evaluating

Run the following command to evaluate Uni-AdaFocus on the validation split of your dataset.
* add --evaluate to turn to the evaluation mode
* add --resume to specify the model checkpoint to evaluate

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main.py \
  --glance_arch mbv2 --dataset YOUR_DATA --data_dir PATH_TO_YOUR_DATASET   \
  --root_log LOG_DIR  \
  --workers 16 --num_segments 48 --dropout 0.2  --fc_dropout 0.2   \
  --epochs 50 --lr 0.002 --lr_type cos --lr_steps 0 0   \
  --batch_size 32 --momentum 0.9 --weight_decay 2e-4   \
  --patch_size 128 --global_lr_ratio 0.5 --stn_lr_ratio 0.2 --classifier_lr_ratio 20.0 --temporal_lr_ratio 0.2 --KL_ratio 0.0 --norm_ratio 1.00  \
  --hidden_dim 1024 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
  --num_glance_segments 16 --num_focus_segments 16 --temperature 1.0 \
  --evaluate \
  --resume LOG_DIR/MODEL_NAME/ckpt.best.pth.tar
```
