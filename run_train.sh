#!/bin/bash

v_format_data_path=../KITTI-FORMAT/
v_checkpoint_dir=checkpoints/
v_init_checkpoint=checkpoints/model.latest
v_img_width=416
v_img_height=128
v_batch_size=4

# python gen_seg_seq.py
# python train.py --dataset_dir=$v_format_data_path --checkpoint_dir=$v_checkpoint_dir --img_width=$v_img_width --img_height=$v_img_height --batch_size=$v_batch_size
# -m pdb
python train.py --dataset_dir=$v_format_data_path --init_checkpoint_file=$v_init_checkpoint --checkpoint_dir=$v_checkpoint_dir --img_width=$v_img_width --img_height=$v_img_height --batch_size=$v_batch_size
# -m pdb
