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
from depth2normal_tf import *
from normal2depth_tf import *

class SfMLearner(object):
    def __init__(self):
        pass
    def gradient(self, pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy
    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
        with tf.name_scope("data_loading"):
            # Load batch data
            # tgt_image[batch;img_height, img_width, 3]
            # seg_image[batch; image_height, image_width, 1]
            # src_image[batch;img_height,img_width, num_source * 3]
            # intrinsics [batch; 3; 3] 

            tgt_image, seg_img, src_image_stack, seg_image_stack, intrinsics = loader.load_train_batch()
            tgt_image = self.preprocess_image(tgt_image)
            src_image_stack = self.preprocess_image(src_image_stack)

        with tf.name_scope("depth_prediction"):
            if(self.opt.background_only):
                tgt_image_fg, tgt_image_bg, foreground_msk = maskout_partial_image(tgt_image, seg_img, target="both")
                src_image_stack_fore, src_image_stack_bg = maskout_partial_image(src_image_stack, seg_image_stack, target="stack")
                
                pred_disp_foreground, _ = disp_net(tgt_image_fg, is_training=True, scope_name="fore")
                pred_depth_fore = [1./d for d in pred_disp_foreground]

            pred_disp, depth_net_endpoints = disp_net(tgt_image, 
                                                      is_training=True)
            pred_depth = [1./d for d in pred_disp]

        with tf.name_scope("pose_and_explainability_prediction"):
            if(self.opt.background_only):
                pred_poses, pred_exp_logits, pose_exp_net_endpoints = \
                    pose_exp_net(tgt_image_bg,
                                src_image_stack_bg, 
                                do_exp=(opt.explain_reg_weight > 0),
                                is_training=True)
            else:
                pred_poses, pred_exp_logits, pose_exp_net_endpoints = \
                    pose_exp_net(tgt_image,
                    src_image_stack, 
                    do_exp=(opt.explain_reg_weight > 0),
                    is_training=True)    
        with tf.name_scope('get_edge_mask'):
            if(opt.edge_mask_weight > 0):
                edge_image = edge_net(seg_img)
        with tf.name_scope("compute_loss"):
            pixel_loss = 0
            exp_loss = 0
            smooth_loss = 0

            # more loss
            img_grad_loss = 0
            normal_smooth_loss = 0
            depth_from_normal_smooth = 0
            edge_loss = 0

            # Depth-Normal Constrains
            pred_normals = []

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

                if opt.background_only:
                    # pred_depth_fore_tensor = tf.expand_dims(tf.squeeze(pred_depth_fore[s], axis=3), -1)
                    pred_depth_fore_tensor = tf.squeeze(pred_depth_fore[s], axis=3)
                    curr_src_image_stack_fore = tf.image.resize_area(src_image_stack_fore, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

                    curr_tgt_image_fore = tf.image.resize_area(tgt_image_fg,\
                                        [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
                    smooth_loss+= opt.fore_weight * opt.smooth_weight/(2**s) * \
                        self.compute_smooth_loss(pred_disp_foreground[s])

                pred_depth_tensor = tf.squeeze(pred_depth[s], axis=3)

                if opt.smooth_weight > 0:
                    smooth_loss += opt.smooth_weight/(2**s) * \
                        self.compute_smooth_loss(pred_disp[s])
                if opt.normal_weight > 0 or opt.dnrefine_weight>0:
                    cam_mat = intrinsics[:,s,:,:] #[batch;3;3]
                    # just extract fx, fy, cx, cy of each intrinsic[batch; 4]
                    intrinsic_params = tf.concat([tf.expand_dims(cam_mat[:,0,0],1),\
                                              tf.expand_dims(cam_mat[:,1,1],1), \
                                              tf.expand_dims(cam_mat[:,0,2],1), \
                                              tf.expand_dims(cam_mat[:,1,2],1)], 1)

                    pred_normal = depth2normal_layer_batch(pred_depth_tensor, intrinsic_params, False)
                    if(opt.dnrefine_weight > 0):
                        pred_depth_from_normal = normal2depth_layer_batch(pred_depth_tensor, tf.squeeze(pred_normal), intrinsic_params, curr_tgt_image)
                        pred_depth_from_normal = tf.expand_dims(pred_depth_from_normal, -1)
                        depth_from_normal_smooth += opt.dnrefine_weight/(2**s) * \
                            self.compute_smooth_loss(pred_depth_from_normal)
                        # pred_disp_from_normal = 1.0 / pred_depth_from_normal
                        # pred_normals.append(pred_normal)
                        # pred_disp_from_norm_stack.append(pred_disp_from_normal)
                    if(opt.normal_weight > 0):
                        normal_smooth_loss+=opt.smooth_weight/(2**s) * \
                            self.compute_smooth_loss(pred_normal)
                
                    
                for i in range(opt.num_source):
                    # Inverse warp the source image to the target image frame
                    curr_proj_image = projective_inverse_warp(
                        curr_src_image_stack[:,:,:,3*i:3*(i+1)], 
                        pred_depth_tensor, 
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
                    if opt.background_only:
                        curr_proj_image_fore = projective_inverse_warp(
                                                curr_src_image_stack_fore[:,:,:,3*i:3*(i+1)], 
                                                pred_depth_fore_tensor, 
                                                pred_poses[:,i,:], 
                                                intrinsics[:,s,:,:])
                        curr_proj_error_fore = tf.abs(curr_proj_image_fore - curr_tgt_image_fore)
                        pixel_loss += opt.fore_weight * tf.reduce_mean(curr_proj_error_fore)
                    
                    if opt.edge_mask_weight > 0:
                        ref_edge_mask = self.get_reference_explain_mask(s)[:,:,:,0]
                        curr_edge_img = edge_image[s]

                        edge_loss += tf.multiply(opt.edge_mask_weight/(2**s), \
                            tf.reduce_mean(tf.square(tf.squeeze(curr_edge_img)-ref_edge_mask)))

                    # Structure Similarity
                    if opt.ssim_weight > 0:
                        pixel_loss += opt.ssim_weight * tf.reduce_mean(SSIM(curr_proj_image, curr_tgt_image))
                        
                    # Gradient Loss
                    if opt.img_grad_weight > 0:
                        curr_tgt_image_grad_x, curr_tgt_image_grad_y = self.gradient(curr_tgt_image[:, :-2, 1:-1, :])
                        curr_src_image_grad_x, curr_src_image_grad_y = self.gradient(curr_src_image_stack[:, :-2, 1:-1 :])

                        curr_proj_image_grad_x, curr_proj_image_grad_y = self.gradient(curr_proj_image[:, :-2, 1:-1, :])
                        curr_proj_error_grad_x, curr_proj_error_grad_y = tf.abs(curr_tgt_image_grad_x-curr_proj_image_grad_x), \
                                                                tf.abs(curr_tgt_image_grad_y-curr_proj_image_grad_y)

                        img_grad_loss += opt.img_grad_weight * tf.reduce_mean(curr_proj_error_grad_x)
                        img_grad_loss += opt.img_grad_weight * tf.reduce_mean(curr_proj_error_grad_y)

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
            total_loss = pixel_loss + smooth_loss + exp_loss + img_grad_loss + normal_smooth_loss+depth_from_normal_smooth + edge_loss

        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            
            if(opt.training_decay):
                learning_rate = tf.train.exponential_decay(
                                opt.learning_rate,   # Base learning rate.
                                tf.Variable(0) * opt.batch_size,  # Current index into the dataset.
                                5000,                # Decay step.
                                0.95,                # Decay rate.
                                staircase=True)
            else:
                learning_rate = opt.learning_rate
            optim = tf.train.AdamOptimizer(learning_rate, opt.beta1)
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
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        opt.num_scales = 4
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
            # print('Trainable variables: ')
            # for var in tf.trainable_variables():
            #     print(var.name)
            # print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                sess.graph._unsafe_unfinalize()
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                # self.saver.restore(sess, checkpoint)
                optimistic_restore(sess, checkpoint)
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
    def build_normal_test_graph(self, pred_mode='direct'):
        input_uint8 = tf.placeholder(tf.uint8, \
                                    [self.batch_size, self.img_height, self.img_width, 3],\
                                    name='raw_input')
        input_mc = self.preprocess_image(input_uint8)

        gt_depth_image = tf.placeholder(tf.uint8,\
                                    [self.batch_size, self.img_height,self.img_width],\
                                    name="gt_depth")
        gt_depth_tensor = self.preprocess_image(gt_depth_image)

        intrinsics = tf.placeholder(tf.float32,\
                                    [self.batch_size, 3, 3],\
                                    name="intrinsic")

        cam_mat = intrinsics[:,:,:] #[batch;3;3]
        # just extract fx, fy, cx, cy of each intrinsic[batch; 4]
        intrinsic_params = tf.concat([tf.expand_dims(cam_mat[:,0,0],1),\
                                    tf.expand_dims(cam_mat[:,1,1],1), \
                                    tf.expand_dims(cam_mat[:,0,2],1), \
                                    tf.expand_dims(cam_mat[:,1,2],1)], 1)

        gt_norm = depth2normal_layer_batch(gt_depth_tensor, intrinsic_params, False)

        if(pred_mode == 'direct'):
            with tf.name_scope("normal_prediction"):
                #Ted DO STH!!
                pred_norm, depth_net_endpoints = disp_net(input_mc, is_training=False)
                pred_norm = pred_norm[0]
        else:
            with tf.name_scope("normal_from_depth"):
                pred_disp, depth_net_endpoints = disp_net(input_mc, is_training=False)
                pred_depth = [1./disp for disp in pred_disp]
                pred_depth_tensor = tf.squeeze(pred_depth[0], axis=3)
                pred_norm = depth2normal_layer_batch(pred_depth_tensor, intrinsic_params, False)
                
        self.inputs = input_uint8
        self.intrinsic = intrinsics
        self.gtdepth = gt_depth_image
        self.pred_norm = pred_norm
        self.gt_norm = gt_norm
        self.pred_depth = pred_depth[0]
        
    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
                    self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        # if(opt.background_only):
        #     input_uint8_seg = tf.placeholder(tf.uint8, [self.batch_size, 
        #                 self.img_height, self.img_width, 3], name='raw_input_seg')
        #     input_mc_seg = self.preprocess_image(input_uint8_seg)


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
        if self.mode == 'norm':
            self.build_normal_test_graph(pred_mode="indirect")

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses            
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results
    def inference_normal(self, rgb, depth, intrinsic, sess):
        fetches = {}
        fetches['normal'] = self.pred_norm
        fetches['gtnormal'] = self.gt_norm
        fetches['pdepth'] =  self.pred_depth 
        results = sess.run(fetches, feed_dict={self.inputs:rgb, self.gtdepth:depth,self.intrinsic:intrinsic})
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
