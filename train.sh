CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py --script aqatrack --config AQATrack-ep150-full-256 --save_dir ./output --mode multiple --nproc_per_node 4
