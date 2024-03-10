# OSTrack
The official implementation for the **CVPR 2024** paper [_Autoregressive Queries for Adaptive Tracking with Spatio-Temporal Transformers_](https://arxiv.org/abs/2203.11991) 

Models:[[Models]](https://drive.google.com/drive/folders/1nfdoPeqah2qYnHuazQMdYFKsrwL5eBEM)
Raw Results:[[Raw Results]](https://drive.google.com/drive/folders/1nfdoPeqah2qYnHuazQMdYFKsrwL5eBEM)
Training logs:[[Training logs]](https://drive.google.com/drive/folders/1nfdoPeqah2qYnHuazQMdYFKsrwL5eBEM)

<!-- <p align="center">
  <img width="85%" src="https://github.com/botaoye/OSTrack/blob/main/assets/arch.png" alt="Framework"/>
</p> -->


## :sunny: Highlights

### :star2: New One-stream Tracking Framework
OSTrack is a simple, neat, high-performance **one-stream tracking framework** for joint feature learning and relational modeling based on self-attention operators.
Without any additional temporal information, OSTrack achieves SOTA performance on multiple benchmarks. OSTrack can serve as a strong baseline for further research.

| Tracker     | LaSOT (AUC)|LaSOT<sub>ext (AUC)|UAV123 (AUC)|TrackingNet (AUC)|GOT-10K (AO)
|:-----------:|:----------:|:-----------------:|:----------:|:---------------:|:----------:
| AQATrack-256 | 71.4      | 51.2              | 70.7       | 83.8            | 73.8         
| AQATrack-384 | 72.7      | 52.7              | 71.2       | 84.8            | 76.0         


## Install the environment
**Option1**: Use the Anaconda
```
conda create -n aqatrack python=3.8
conda activate aqatrack
bash install.sh
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training
Download pre-trained [HiViT-Base weights](https://drive.google.com/file/d/1VZQz4buhlepZ5akTcEvrA3a_nxsQZ8eQ/view?usp=share_link) and put it under `$PROJECT_ROOT$/pretrained_models` (see [HiViT](https://github.com/zhangxiaosong18/hivit) for more details).

```
bash train.sh
```


## Test
```
python test_epoch.py
```

## Evaluation 
```
python tracking/analysis_results.py
```


## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX2080Ti GPU.

```
# Profiling AQATrack-ep150-full-256
python tracking/profile_model.py --script aqatrack --config AQATrack-ep150-full-256
# Profiling AQATrack-ep150-full-384
python tracking/profile_model.py --script aqatrack --config AQATrack-ep150-full-384
```


## Acknowledgments
* Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library, which helps us to quickly implement our ideas.


## Citation
If our work is useful for your research, please consider cite:

```
@article{xie2024aqatrack,
  title={Autoregressive Queries for Adaptive Tracking with Spatio-Temporal Transformers},
  author={},
  journal={},
  year={2024}
}
```
