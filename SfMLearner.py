from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from nets import *
from utils import *

class SfMLearner(object):
    def __init__(self):
        pass

    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
        with tf.name_scope("data_loading"):
            tgt_image, src_image_stack, intrinsics = loader.load_train_batch()
            tgt_image = self.preprocess_image(tgt_image)
            src_image_stack = self.preprocess_image(src_image_stack)

        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(tgt_image,
                                                      is_training=True)
            pred_depth = [1./d for d in pred_disp]

        # added normal prediction
        with tf.name_scope("normal_prediction"):
            pred_norm, normal_net_endpoints = normal_net(tgt_image,
                                                         is_training=True)

        with tf.name_scope("pose_and_explainability_prediction"):
            pred_poses, pred_exp_logits, pose_exp_net_endpoints = \
                pose_exp_net(tgt_image,
                             src_image_stack,
                             do_exp=(opt.explain_reg_weight > 0),
                             is_training=True)

        with tf.name_scope("compute_loss"):
            pixel_loss = 0
            exp_loss = 0
            smooth_loss = 0
            tgt_image_all = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            exp_mask_stack_all = []
            for s in range(opt.num_scales):
                if opt.explain_reg_weight > 0:
                    # Construct a reference explainability mask (i.e. all
                    # pixels are explainable)
                    ref_exp_mask = self.get_reference_explain_mask(s)
                # Scale the source and target images for computing loss at the
                # according scale.
                curr_tgt_image = tf.image.resize_area(tgt_image,
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
                curr_src_image_stack = tf.image.resize_area(src_image_stack,
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

                if opt.smooth_weight > 0:
                    smooth_loss += opt.smooth_weight/(2**s) * \
                        self.compute_smooth_loss(pred_disp[s])

                for i in range(opt.num_source):
                    # Inverse warp the source image to the target image frame
                    curr_proj_image = projective_inverse_warp(
                        curr_src_image_stack[:,:,:,3*i:3*(i+1)],
                        tf.squeeze(pred_depth[s], axis=3),
                        pred_poses[:,i,:],
                        intrinsics[:,s,:,:])
                    curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                    # Cross-entropy loss as regularization for the
                    # explainability prediction
                    if opt.explain_reg_weight > 0:
                        curr_exp_logits = tf.slice(pred_exp_logits[s],
                                                   [0, 0, 0, i*2],
                                                   [-1, -1, -1, 2])
                        exp_loss += opt.explain_reg_weight * \
                            self.compute_exp_reg_loss(curr_exp_logits,
                                                      ref_exp_mask)
                        curr_exp = tf.nn.softmax(curr_exp_logits)
                    # Photo-consistency loss weighted by explainability
                    if opt.explain_reg_weight > 0:
                        pixel_loss += tf.reduce_mean(curr_proj_error * \
                            tf.expand_dims(curr_exp[:,:,:,1], -1))
                    else:
                        pixel_loss += tf.reduce_mean(curr_proj_error)
                    # Prepare images for tensorboard summaries
                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,1], -1)
                    else:
                        proj_image_stack = tf.concat([proj_image_stack,
                                                      curr_proj_image], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack,
                                                      curr_proj_error], axis=3)
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.concat([exp_mask_stack,
                                tf.expand_dims(curr_exp[:,:,:,1], -1)], axis=3)
                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                if opt.explain_reg_weight > 0:
                    exp_mask_stack_all.append(exp_mask_stack)
            total_loss = pixel_loss + smooth_loss + exp_loss

        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
            # self.grads_and_vars = optim.compute_gradients(total_loss,
            #                                               var_list=train_vars)
            # self.train_op = optim.apply_gradients(self.grads_and_vars)
            self.train_op = slim.learning.create_train_op(total_loss, optim)
            self.global_step = tf.Variable(0,
                                           name='global_step',
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step,
                                              self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_depth = pred_depth
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.total_loss = total_loss
        self.pixel_loss = pixel_loss
        self.exp_loss = exp_loss
        self.smooth_loss = smooth_loss
        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        self.exp_mask_stack_all = exp_mask_stack_all

    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp,
                               (opt.batch_size,
                                int(opt.img_height/(2**downscaling)),
                                int(opt.img_width/(2**downscaling)),
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask

    def compute_exp_reg_loss(self, pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)

    def compute_smooth_loss(self, pred_disp):
        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy
        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return tf.reduce_mean(tf.abs(dx2)) + \
               tf.reduce_mean(tf.abs(dxdy)) + \
               tf.reduce_mean(tf.abs(dydx)) + \
               tf.reduce_mean(tf.abs(dy2))

    def compute_normal_depth_loss(self, img, pred_disp, pred_norm,
                                    alpha=0.1, n_shift=3, inverse_depth=True):
        """Compute the dot product of normals and the 8-neighbor vectors

        Args:
            img: target RGB image [batch, H, W, 3]
            pred_disp: predicted depth or inverse depth [batch, H, W, 1]
            pred_norm: predicted surface normals [batch, H, W, 3]
            alpha: hyperparameter used to weight 8-neighbor vectors by pixel intensity difference
            n_shift: pixels to shift when calculating 8-neighbor vectors
            inverse_depth: set to `True` if the input `pred_disp` is inverse depth

        Returns:
            The sum of dot product of normals and the 8-neighbor vectors for each pixel
        """

        batch, height, width, _ = pred_disp.get_shape().as_list()
        mask = tf.greater(pred_disp, tf.zeros_like(pred_disp)) # mask indicating nonzero depth values
        mask = tf.cast(mask, tf.float32)
        depth_map = tf.identity(pred_disp) # a copy of `pred_disp`

        # convert inverse depth to depth by assigning `eps` to every pixel with 0 inverse depth
        # then inversing the inverse depth map
        if inverse_depth:
            eps = 1e-8
            depth_eps = eps * (1.0 - mask)
            depth_map += depth_eps
            depth_map = 1 / depth_map

        # get 3D coordinates of points
        # done as in `projective_inverse_warp`
        pixel_coords = meshgrid(batch, height, width)
        cam_coords = pixel2cam(depth_map, pixel_coords, intrinsics, is_homogeneous=False) #[batch, 3, height, width]
        # note that `cam_coords` has shape [batch, 3, height, width]
        img = tf.transpose(cam_coords, perm=[0, 2, 3, 1]) # [batch, height, width, 3]

        # copied from https://github.com/zhenheny/LEGO/blob/master/depth2normal/depth2normal_tf.py
        nei = n_shift
        # shift the 3d pts map by nei along 8 directions
        pts_3d_map_ctr = pts_3d_map[:, nei:-nei, nei:-nei, :]
        pts_3d_map_x0 = pts_3d_map[:, nei:-nei, 0:-(2*nei), :]
        pts_3d_map_y0 = pts_3d_map[:, 0:-(2*nei), nei:-nei, :]
        pts_3d_map_x1 = pts_3d_map[:, nei:-nei, 2*nei:, :]
        pts_3d_map_y1 = pts_3d_map[:, 2*nei:, nei:-nei, :]
        pts_3d_map_x0y0 = pts_3d_map[:, 0:-(2*nei), 0:-(2*nei), :]
        pts_3d_map_x0y1 = pts_3d_map[:, 2*nei:, 0:-(2*nei), :]
        pts_3d_map_x1y0 = pts_3d_map[:, 0:-(2*nei), 2*nei:, :]
        pts_3d_map_x1y1 = pts_3d_map[:, 2*nei:, 2*nei:, :]

        # generate difference between the central pixel and one of 8 neighboring pixels
        # each `diff` has shape [batch, H-2*nei, W-2*nei, 3]
        diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
        diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
        diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
        diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
        diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
        diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
        diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
        diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

        # pad each `diff` with zeros to have [batch, H, W, 3]
        diff_x0_pad = tf.pad(diff_x0, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        diff_y0_pad = tf.pad(diff_y0, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        diff_x1_pad = tf.pad(diff_x1, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        diff_y1_pad = tf.pad(diff_y1, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        diff_x0y0_pad = tf.pad(diff_x0y0, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        diff_x0y1_pad = tf.pad(diff_x0y1, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        diff_x1y0_pad = tf.pad(diff_x1y0, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        diff_x1y1_pad = tf.pad(diff_x1y1, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        diff_pad = tf.stack([diff_x0_pad, diff_y0_pad, diff_x1_pad, diff_y1_pad,
                             diff_x0y0_pad, diff_x0y1_pad, diff_x1y0_pad, diff_x1y1_pad], axis=0) # [8, batch, H, W, 3]
        print('diff_pad.shape', diff_pad.get_shape().as_list())

        # generate weights for 8-neighbor vectors
        img_ctr = img[:, nei:-nei, nei:-nei, :]
        img_x0 = img[:, nei:-nei, 0:-(2*nei), :]
        img_y0 = img[:, 0:-(2*nei), nei:-nei, :]
        img_x1 = img[:, nei:-nei, 2*nei:, :]
        img_y1 = img[:, 2*nei:, nei:-nei, :]
        img_x0y0 = img[:, 0:-(2*nei), 0:-(2*nei), :]
        img_x0y1 = img[:, 2*nei:, 0:-(2*nei), :]
        img_x1y0 = img[:, 0:-(2*nei), 2*nei:, :]
        img_x1y1 = img[:, 2*nei:, 2*nei:, :]

        grad_x0_pad = tf.pad(img_x0 - img_ctr, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        grad_y0_pad = tf.pad(img_y0 - img_ctr, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        grad_x1_pad = tf.pad(img_x1 - img_ctr, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        grad_y1_pad = tf.pad(img_y1 - img_ctr, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        grad_x0y0_pad = tf.pad(img_x0y0 - img_ctr, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        grad_x0y1_pad = tf.pad(img_x0y1 - img_ctr, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        grad_x1y0_pad = tf.pad(img_x1y0 - img_ctr, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
        grad_x1y1_pad = tf.pad(img_x1y1 - img_ctr, [[0, 0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")

        w_x0 = tf.exp(-alpha * tf.reduce_mean(tf.abs(grad_x0_pad), axis=3))
        w_y0 = tf.exp(-alpha * tf.reduce_mean(tf.abs(grad_y0_pad), axis=3))
        w_x1 = tf.exp(-alpha * tf.reduce_mean(tf.abs(grad_x1_pad), axis=3))
        w_y1 = tf.exp(-alpha * tf.reduce_mean(tf.abs(grad_y1_pad), axis=3))
        w_x0y0 = tf.exp(-alpha * tf.reduce_mean(tf.abs(grad_x0y0_pad), axis=3))
        w_x0y1 = tf.exp(-alpha * tf.reduce_mean(tf.abs(grad_x0y1_pad), axis=3))
        w_x1y0 = tf.exp(-alpha * tf.reduce_mean(tf.abs(grad_x1y0_pad), axis=3))
        w_x1y1 = tf.exp(-alpha * tf.reduce_mean(tf.abs(grad_x1y1_pad), axis=3))

        w = tf.stack([w_x0, w_y0, w_x1, w_y1, w_x0y0, w_x0y1, w_x1y0, w_x1y1], axis=0) # [8, batch, H, W]
        print('w.shape: ', w.get_shape().as_list())
        w = w / tf.reduce_sum(w, axis=0) # normalize weights for each pixel, so they sum to 1

        # calculate loss by dot product
        loss = 0
        for v in range(8):
            element_wise_prod = tf.multiply(diff_pad[v, :, :, :, :], pred_norm[:, :, :, :]) # [batch, H, W, 3]
            dot_prod = tf.reduce_sum(element_wise_prod, axis=3)
            loss += tf.multiply(w[v, :, :, :], dot_prod)

        return loss

    def compute_normal_reg_loss(self, pred_norm):
        # compute norm of surface normals
        norm = tf.norm(
                    pred_norm,
                    ord='euclidean',
                    axis=3,
                    name='normal_norm')

        ones = tf.ones_like(norm)

        # compute squared error between norm and ones
        loss = tf.reduce_mean((norm - ones) ** 2)
        return loss

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("exp_loss", self.exp_loss)
        for s in range(opt.num_scales):
            tf.summary.histogram("scale%d_depth" % s, self.pred_depth[s])
            tf.summary.image('scale%d_disparity_image' % s, 1./self.pred_depth[s])
            tf.summary.image('scale%d_target_image' % s, \
                             self.deprocess_image(self.tgt_image_all[s]))
            for i in range(opt.num_source):
                if opt.explain_reg_weight > 0:
                    tf.summary.image(
                        'scale%d_exp_mask_%d' % (s, i),
                        tf.expand_dims(self.exp_mask_stack_all[s][:,:,:,i], -1))
                tf.summary.image(
                    'scale%d_source_image_%d' % (s, i),
                    self.deprocess_image(self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_projected_image_%d' % (s, i),
                    self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_proj_error_%d' % (s, i),
                    self.deprocess_image(tf.clip_by_value(self.proj_error_stack_all[s][:,:,:,i*3:(i+1)*3] - 1, -1, 1)))
        tf.summary.histogram("tx", self.pred_poses[:,:,0])
        tf.summary.histogram("ty", self.pred_poses[:,:,1])
        tf.summary.histogram("tz", self.pred_poses[:,:,2])
        tf.summary.histogram("rx", self.pred_poses[:,:,3])
        tf.summary.histogram("ry", self.pred_poses[:,:,4])
        tf.summary.histogram("rz", self.pred_poses[:,:,5])
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        # for grad, var in self.grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)

    def train(self, opt):
        #opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        #opt.num_scales = 4
        self.opt = opt
        self.build_train_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=10)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(1, opt.max_steps):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }

                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/opt.summary_freq,
                                results["loss"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                    self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(
                input_mc, is_training=False)
            pred_depth = [1./disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth
        self.depth_epts = depth_net_endpoints

    def build_pose_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
            self.img_height, self.img_width * self.seq_length, 3],
            name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _, _ = pose_exp_net(
                tgt_image, src_image_stack, do_exp=False, is_training=False)
            self.inputs = input_uint8
            self.pred_poses = pred_poses

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self,
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
