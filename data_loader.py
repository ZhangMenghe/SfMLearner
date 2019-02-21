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
                 num_scales=None,
                 num_seg_class=None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales
        self.num_images = num_source + 1
        self.num_seg_class = num_seg_class

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, 'train')

        self.steps_per_epoch = int(
	        len(file_list['image_file_list'])//self.batch_size)
        
        buffer_size = tf.shape(file_list['seg_file_list'], out_type=tf.int64)[0]
        dataset = tf.data.Dataset.from_tensor_slices(( 
            file_list['seg_file_list'], 
            file_list['image_file_list'], 
            file_list['cam_file_list'])).shuffle(buffer_size, seed = seed)

        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        
        def read_image(filename, is_seg = False):
            img_string = tf.read_file(filename)
            decode = tf.image.decode_png(img_string)
            tgt_img, src_stack = self.unpack_image_sequence(
                decode, 
                self.img_height,
                self.img_width, 
                self.num_source, 
                is_seg)
            return tgt_img, src_stack
        
        def read_cam(filename):
            raw_cam_contents = tf.read_file(filename)
            raw_cam_vec = tf.decode_csv(raw_cam_contents, record_defaults=rec_def)
            raw_cam_vec = tf.stack(raw_cam_vec)
            intrinsics = tf.reshape(raw_cam_vec, [3, 3])
            return intrinsics
        
        def _parse_function(seg_filename, img_filename, cam_filename):
            img_sequence = read_image(img_filename)
            seg_sequence = read_image(seg_filename, True)
            intrinsics = read_cam(cam_filename)
            return img_sequence, seg_sequence, intrinsics

        dataset = dataset.map(_parse_function)
        batch = dataset.batch(self.batch_size)
        iterator = batch.make_one_shot_iterator()
        
        img_sequence, seg_sequence, intrinsics = iterator.get_next()
        tgt_image, src_image_stack, tgt_seg, src_seg_stack = \
            img_sequence[0], img_sequence[1], seg_sequence[0], seg_sequence[1]

        seg_stack = tf.concat([tgt_seg, src_seg_stack], axis=3) #[4, H, W, 3]
        img_stack = tf.concat([tgt_image, src_image_stack], axis=3) #[4, H, W, 9]
        
        image_all = tf.concat([img_stack, seg_stack], axis=3) #[4, H, W, 12]
        image_all, intrinsics = self.data_augmentation(
            image_all, intrinsics, self.img_height, self.img_width)
        
        img_depth = 3 * (self.num_source + 1)
        img_stack, seg_stack = image_all[:, :, :, :img_depth], image_all[:, :, :, img_depth:]
        tgt_seg, src_seg_stack = self.unpack_mask_stack(seg_stack) 
        
        tgt_image = tf.concat([img_stack[:, :, :, :3], tgt_seg], axis=3)
        src_image_stack = tf.concat([img_stack[:, :, :, 3:], src_seg_stack], axis=3)
        
        image_all = tf.concat([tgt_image, src_image_stack], axis = 3)
        
        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)

        return tgt_image, src_image_stack, intrinsics

    
    


    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [self.batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w):
        # Random scaling
        def random_scaling(im, intrinsics):
            _, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            ## resize img and segmentation masks 
            img_stack = tf.image.resize_area(im[:, :, :, : 3 * self.num_images], [out_h, out_w])
            seg_stack = tf.image.resize_images(im[:, :, :, 3 * self.num_images:], [out_h, out_w], \
                                               method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            seg_stack = tf.cast(seg_stack, dtype=tf.float32)
            im = tf.concat([img_stack, seg_stack], axis=3)
            
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            _, in_h, in_w, _ = tf.unstack(tf.shape(im))
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
            frame_ids[i] + '.png') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        seg_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '_seg.png') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        all_list['seg_file_list'] = seg_file_list
        return all_list

    def unpack_mask_stack(self, seg_stack):
        tgt_seg = self.unpack_mask(seg_stack[:, :, :, 0])
        src_seg_stack = []
        for i in range(1, self.num_images):
            src_seg_stack.append(self.unpack_mask(seg_stack[:, :, :, i]))
        src_seg_stack = tf.concat(src_seg_stack, axis = 3)
        src_seg_stack = tf.squeeze(src_seg_stack)
        src_seg_stack.set_shape([self.batch_size, 
                                self.img_height, 
                                self.img_width, 
                                self.num_seg_class * self.num_source])
        tgt_seg.set_shape([self.batch_size, self.img_height, self.img_width, self.num_seg_class])
        return tgt_seg, src_seg_stack

    def unpack_mask(self, img):
        n = tf.ones_like(img, dtype = tf.uint8) * 255
        mask = [tf.cast(tf.math.equal(img, n), tf.uint8)]
        for i in range(self.num_seg_class - 1):
            n = tf.ones_like(img, dtype = tf.uint8) * i
            mask.append(tf.cast(tf.math.equal(img, n), tf.uint8))
        mask = tf.stack(mask, axis = 3)
        mask = tf.squeeze(mask)
        return mask


    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source, isMask = False):
        d = 3
        if isMask:
            d = 1
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
                                   num_source * d])
        tgt_image.set_shape([img_height, img_width, d])
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