import tensorflow as tf
from depth2normal_tf import *
from normal2depth_tf import *
import numpy as np
from kitti_eval.depth_evaluation_utils import *
from SfMLearner import SfMLearner
import PIL.Image as pil

flags = tf.app.flags
flags.DEFINE_string("test_idx", 'data/kitti/test_files_eigen.txt', "choose files for test")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string("pred_filename", None, "name of prediction file")
FLAGS = flags.FLAGS

def main(_):
    gt_path = FLAGS.dataset_dir
    basename = FLAGS.pred_filename + '.npy'
    output_pred_norm_file = FLAGS.output_dir + '/' + basename
    output_gt_norm_file = FLAGS.output_dir + '/' + FLAGS.pred_filename+'-gt.npy'
    test_files = read_text_lines(FLAGS.test_idx)
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, gt_path)

    # load gt depth
    gt_depths = []
    gt_intrinsics = []
    num_test = len(im_files)
    for t_id in range(num_test):
        camera_id = cams[t_id]
        depth = generate_depth_map(gt_calib[t_id], 
                                gt_files[t_id], 
                                im_sizes[t_id], 
                                camera_id, 
                                False, 
                                True)
        gt_depths.append(depth.astype(np.float32))
        
        cam2cam = read_calib_file(gt_calib[t_id] + 'calib_cam_to_cam.txt')
        P_rect = cam2cam['P_rect_0'+str(camera_id)].reshape(3,4)
        intrinsics = P_rect[:3, :3]
        gt_intrinsics.append(intrinsics)
    # construct inference graph
    sfm = SfMLearner()
    sfm.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        batch_size=FLAGS.batch_size,
                        mode='norm')
    saver = tf.train.Saver([var for var in tf.model_variables()]) 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        pred_norms = []
        gt_norms =[]
        for t in range(0, len(test_files),FLAGS.batch_size):
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, len(test_files)))
            inputs = np.zeros((FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3), dtype=np.uint8)
            input_depths = np.zeros((FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width), dtype=np.uint8)
            input_intrinsics= np.zeros((FLAGS.batch_size, 3, 3),dtype=np.float32)
            
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                raw_im = pil.open(gt_path + test_files[idx])
                scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                inputs[b] = np.array(scaled_im)
                
                # convert values to 0 - 255 int8 format
                formatted = (gt_depths[idx] * 255 / np.max(gt_depths[idx])).astype('uint8')
                gt_depth_img = pil.fromarray(formatted)
                
                input_depths[b] = np.array(gt_depth_img.resize((FLAGS.img_width, FLAGS.img_height),\
                                                            pil.NEAREST))                
                input_intrinsics[b] = gt_intrinsics[idx]

            pred = sfm.inference_normal(inputs, input_depths, input_intrinsics, sess)
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_norms.append(pred['normal'][b])
                gt_norms.append(pred['gtnormal'][b])
                #pred['pdepth']
        np.save(output_pred_norm_file, pred_norms)
        np.save(output_gt_norm_file, gt_norms)

if __name__ == '__main__':
    tf.app.run()