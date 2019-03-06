model_name='model.latest'
# model_name='model-191178'
pred_file_name='results/'$model_name'.npy'
# pred_file_name='kitti_eval/kitti_eigen_depth_predictions.npy'

if [ ! -f $pred_file_name ]; then
	python test_kitti_depth.py --dataset_dir="../raw_data_downloader/kitti-val/" --output_dir="./results/" --ckpt_file="checkpoints/$model_name" --test_idx="data/kitti/test_files_eigen.txt"
fi
python kitti_eval/eval_depth.py --kitti_dir='../raw_data_downloader/kitti-val/' --pred_file=$pred_file_name