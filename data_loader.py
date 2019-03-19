from __future__ import division
import os
import random
import tensorflow as tf

class DataLoader(object):
    def __init__(self, 
                 dataset_dir=None, 
                 batch_size=None, 
                 img_height=None, 
                 img_width=None, 
                 num_source=None, 
                 num_scales=None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales

    def load_train_batch(self, load_semantic = False, load_semantic_batch = False):
        seed = random.randint(0, 2**31 - 1)

        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, 'train')
        image_path_strs = tf.data.Dataset.from_tensor_slices(file_list['image_file_list'])\
                          .shuffle(len(file_list['image_file_list']), seed = seed)
        cam_path_strs = tf.data.Dataset.from_tensor_slices(file_list['cam_file_list'])\
                        .shuffle(len(file_list['cam_file_list']), seed = seed)
        self.steps_per_epoch = int(len(file_list['image_file_list'])//self.batch_size)

        seg_image = None
        seg_image_stack = None
        img_iterator = image_path_strs.make_initializable_iterator()
        img_element = img_iterator.get_next()

        cam_iterator = cam_path_strs.make_initializable_iterator()
        cam_element = cam_iterator.get_next()

        with tf.Session() as sess:
            # Load images
            sess.run(img_iterator.initializer)
            jpg_file_name = sess.run(img_element)
            if(type(jpg_file_name) == bytes):
                jpg_file_name=jpg_file_name.decode('ASCII')
            image_contents = tf.read_file(jpg_file_name)
            image_seq = tf.image.decode_jpeg(image_contents)
            #tgt_img:[height,width,3]
            #stack:[height,width,3*num]
            tgt_image, src_image_stack = self.unpack_image_sequence(\
                                        image_seq, self.img_height,\
                                         self.img_width, self.num_source)
            #if(load_semantic):
            # png_file_name = jpg_file_name[:-4]+"-seg.png"
            png_file_name = jpg_file_name[:-4]+"-segque.png"
            seg_contents = tf.read_file(png_file_name)
            seg_image = tf.image.decode_png(seg_contents, channels=1)
            # seg_image.set_shape([self.img_height, self.img_width, 1])
            seg_image_queue = tf.cast(seg_image/255*19, tf.uint8)

            seg_image, seg_image_stack = self.unpack_image_sequence(\
                            seg_image_queue, self.img_height,\
                                self.img_width, self.num_source, channels=1)

            # Load camera intrinsics
            sess.run(cam_iterator.initializer)
            cam_contents = tf.read_file(sess.run(cam_element))
            rec_def = []
            for i in range(9):
                rec_def.append([1.])
            raw_cam_vec = tf.stack(tf.decode_csv(cam_contents, record_defaults=rec_def))
            intrinsics = tf.reshape(raw_cam_vec, [3, 3])
        
        # Form training batches
        src_image_stack,seg_image_stack, tgt_image, seg_image, intrinsics = \
                tf.train.batch([src_image_stack, seg_image_stack, tgt_image, seg_image, intrinsics], 
                               batch_size=self.batch_size)
        
        image_all = tf.concat([tgt_image, src_image_stack, seg_image_stack, seg_image], axis=3)
        image_all, intrinsics = self.data_augmentation(
            image_all, intrinsics, self.img_height, self.img_width)
        tgt_image = image_all[:, :, :, :3]
        src_image_stack = image_all[:, :, :, 3:9]
        seg_image_stack = image_all[:, :, :, 9:11]
        seg_image = tf.expand_dims(image_all[:,:,:,-1],axis=-1)
        intrinsics = self.get_multi_scale_intrinsics(intrinsics, self.num_scales)

        return tgt_image, seg_image, src_image_stack, seg_image_stack, intrinsics

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w):
        # Random scaling
        def random_scaling(im, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:,0,0]
            fy = intrinsics[:,1,1]
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics
        im, intrinsics = random_scaling(im, intrinsics)
        im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        return im, intrinsics

    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source, channels = 3):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, tgt_start_idx, 0], 
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, int(tgt_start_idx + img_width), 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, i*img_width, 0], 
                                    [-1, img_width, -1]) 
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height, 
                                   img_width, 
                                   num_source * channels])
        tgt_image.set_shape([img_height, img_width, channels])
        return tgt_image, src_image_stack

    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, 0, tgt_start_idx, 0], 
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0, 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, 0, int(tgt_start_idx + img_width), 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, 0, i*img_width, 0], 
                                    [-1, -1, img_width, -1]) 
                                    for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale