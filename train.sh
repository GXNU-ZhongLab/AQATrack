CUDA_VISIBLE_DEVICES=4,5,6,7 python tracking/train.py --script aqatrack --config multi-hivit-ep150-4frames --save_dir ./output --mode multiple --nproc_per_node 4
#python tracking/test_epoch.py
