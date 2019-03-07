#!/bin/bash
model_name='model.latest'
pred_file_name='results/reserve/model.latest.npy'
# pred_file_name='results/'$1'.npy'
# if [[ -n $2 ]];then
# 	model_name=$2
# fi

# if [ ! -f $pred_file_name ]; then
# 	echo "writing model file to .. "$pred_file_name
# 	python test_kitti_depth.py --dataset_dir="../raw_data_downloader/kitti-val/" --output_dir="./results/" --ckpt_file="checkpoints/$model_name" --test_idx="data/kitti/test_files_eigen.txt" --pred_filename=$1
# fi
python kitti_eval/eval_depth.py --kitti_dir='../raw_data_downloader/kitti-val/' --pred_file=$pred_file_name