# Custom finetune TSM

If you have a customized video classification dataset and would like to finetune Uni-AdaFocus on it, please follow this tutorial step-by-step.

Let's begin with this folder [Uni-AdaFocus-TSM with Experiments on Sth-Sth V1&V2 and Jester](../Uni-AdaFocus-TSM%20with%20Experiments%20on%20Sth-Sth%20V1&V2%20and%20Jester).

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
def return_YOUR_DATA(modality):
    filename_categories = os.path.join(ROOT_DATASET, 'classInd.txt')
    if modality == 'RGB':
        root_data = os.path.join(ROOT_DATASET, 'frames')
        filename_imglist_train = os.path.join(ROOT_DATASET, 'train_split.txt')
        filename_imglist_val = os.path.join(ROOT_DATASET, 'val_split.txt')
        prefix = 'image_{:05d}.jpg'
    else:
        print('no such modality:' + modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
```

2. Change [this line](ops/dataset_config.py#L100) to:
```
dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
               'ucf101': return_ucf101, 'hmdb51': return_hmdb51, 'kinetics': return_kinetics}
↓

dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
               'ucf101': return_ucf101, 'hmdb51': return_hmdb51, 'kinetics': return_kinetics,
               'YOUR_DATA': return_YOUR_DATA}
```

## Start training

Run the following command to train Uni-AdaFocus on the train split of your dataset.

* YOUR_DATA: your dataset name, same to above
* LOG_DIR: path to the directory for saving experimental results
* --patch_size: local patch size (96, 128)
* --num_glance_segments: number of the global frames
* --num_focus_segments: number of the local frames
* --num_input_focus_segments: number of the local frame candidates, recommended to be 3*num_focus_segments

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py YOUR_DATA RGB \
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

## Start evaluating

Run the following command to evaluate Uni-AdaFocus on the validation split of your dataset.
* add --evaluate to turn to the evaluation mode
* add --resume to specify the model checkpoint to evaluate

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py YOUR_DATA RGB \
    --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
    --root_log LOG_DIR \
    --root_model LOG_DIR \
    --arch resnet50 --num_glance_segments 8 --num_input_focus_segments 24 --num_focus_segments 8 \
    --gd 20 -j 16 --lr_steps 0 0 --lr_type cos --epochs 50 \
    --batch-size 64 --lr 0.02 --wd 1e-3 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres --npb --print-freq=50 \
    --patch_size 96 --stn_hidden_dim 128 --temporal_hidden_dim 64 \
    --global_lr_ratio 0.5 --stn_lr_ratio 0.20 --temporal_lr_ratio 0.20 --temperature 1.0  --norm_ratio 0.50
    --evaluate \
    --resume LOG_DIR/MODEL_NAME/ckpt.best.pth.tar
```
