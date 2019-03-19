#!/bin/bash
model_name='model.latest'
groundtruth='kitti_eval/pose_data/'
pred_file_name='results/'$1'.npy'
if [[ -n $2 ]];then
	model_name=$2
fi

if [ ! -f $pred_file_name ]; then
	echo "writing model file to .. "$pred_file_name
	python test_kitti_pose.py --test_seq 9 --dataset_dir="../raw_data_downloader/kitti-val/" --output_dir="./results/" --ckpt_file="checkpoints/$model_name"
fi

# python kitti_eval/eval_pose.py --gtruth_dir=$groundtruth --pred_dir=/directory/of/predicted/trajectory/files/