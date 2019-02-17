#!/bin/bash
# v_date_list=('2011_09_26' '2011_09_28' '2011_09_29' '2011_09_30' '2011_10_03')
v_date_list=('2011_09_30')
v_id_list_26=('0001' '0002'  '0005'  '0009'  '0011'  '0013'  '0014'  '0015'  '0017'  '0018'  '0019'  '0020'  '0022'  '0023'  '0027'  '0028'  '0029'  '0032'  '0035'  '0036'  '0039'  '0046'  '0048'  '0051'  '0052'  '0056'  '0057'  '0059'  '0060'  '0061'  '0064'  '0070'  '0079'  '0084'  '0086'  '0087'  '0091'  '0093'  '0095'  '0096'  '0101'  '0104'  '0106'  '0113'  '0117'  '0119')
v_id_list_28=('0001'  '0002'  '0016'  '0021'  '0034'  '0035'  '0037'  '0038'  '0039'  '0043'  '0045'  '0047'  '0053'  '0054'  '0057'  '0065'  '0066'  '0068'  '0070'  '0071'  '0075'  '0077'  '0078'  '0080'  '0082'  '0086'  '0087'  '0089'  '0090'  '0094'  '0095'  '0096'  '0098'  '0100'  '0102'  '0103'  '0104'  '0106'  '0108'  '0110'  '0113'  '0117'  '0119'  '0121'  '0122'  '0125'  '0126'  '0128'  '0132'  '0134'  '0135'  '0136'  '0138'  '0141'  '0143'  '0145'  '0146'  '0149'  '0153'  '0154'  '0155'  '0156'  '0160'  '0161'  '0162'  '0165'  '0166'  '0167'  '0168'  '0171'  '0174'  '0177'  '0179'  '0183'  '0184'  '0185'  '0186'  '0187'  '0191'  '0192'  '0195'  '0198'  '0199'  '0201'  '0204'  '0205'  '0208'  '0209'  '0214'  '0216'  '0220'  '0222'  '0225')
v_id_list_29=('0004'  '0026'  '0071'  '0108')
v_id_list_30=('0016'  '0018'  '0020'  '0027'  '0028'  '0033'  '0034'  '0072')
v_id_list_03=('0027'  '0034'  '0042'  '0047'  '0058')
current_list=()

v_raw_data_set_path='/home/menghe/Github/raw_data_downloader/'
v_dump_root='/home/menghe/Github/KITTI-FORMAT/'
v_seg_graph='kitti-deeplab/model/frozen_inference_graph.pb'
v_dataset_name='kitti_raw_eigen'
v_collect_segs=0
# v_dataset_name='kitti_odom'
if [ $# -ne 0 ]
	then
	if [ $1=='seg' ]
	then
		v_collect_segs=1
		echo $v_collect_segs
	fi
fi
v_seq_length=3
v_img_width=416
v_img_height=128
v_num_threads=1


for i in ${v_date_list[@]}; do
	# calib_zip_name=$i'_calib.zip'
	# echo "Downloading: "$calib_zip_name
	# wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$calib_zip_name -P $v_raw_data_set_path
 #    unzip -o $v_raw_data_set_path$calib_zip_name -d $v_raw_data_set_path
 #    rm $v_raw_data_set_path$calib_zip_name
 #    if [ ${i:(-2)} == "26" ]
 #    then
 #    	current_list=("${v_id_list_26[@]}")
 #    elif [[ ${i:(-2)} == "28" ]]; then
 #    	current_list=("${v_id_list_28[@]}")
 #    elif [[ ${i:(-2)} == "29" ]]; then
 #    	current_list=("${v_id_list_29[@]}")
 #    elif [[ ${i:(-2)} == "30" ]]; then	
 #    	current_list=("${v_id_list_30[@]}")    	
 #    elif [[ ${i:(-2)} == "03" ]]; then	
 #    	current_list=("${v_id_list_03[@]}")
 #    fi
 #    	for id in ${current_list[@]}; do
 #    		name_str=$i'_drive_'$id
 #    		echo $name_str
 #            shortname=$name_str'_sync.zip'
 #            fullname=$name_str'/'$name_str'_sync.zip'

	# 		echo "Downloading: "$shortname
	#         wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname -P $v_raw_data_set_path
	#         unzip -o $v_raw_data_set_path$shortname -d $v_raw_data_set_path
	#         rm $v_raw_data_set_path$shortname
	# 	done
 	python data/prepare_train_data.py --dataset_dir=$v_raw_data_set_path --dataset_name=$v_dataset_name --dump_root=$v_dump_root --date_spe=$i --seq_length=$v_seq_length --img_width=$v_img_width --img_height=$v_img_height --num_threads=$v_num_threads
 	# if [[ $? != 0 ]];
 	# then
 	# 	break
 	# fi
 	# rm -r $v_raw_data_set_path$i
done
