from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from SfMLearner import SfMLearner
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("smooth_weight", 0.5, "Weight for depth map smoothness")
flags.DEFINE_float("normal_weight", 0.0, "Weight for normal map smoothness")
flags.DEFINE_float("dnrefine_weight", 0.3, "Weight for depth/normal refinement constrain")

flags.DEFINE_float("ssim_weight", 0.0, "Weight for using ssim loss in pixel loss")
flags.DEFINE_float("img_grad_weight", 0.0, "Weight for image gradient warping")
flags.DEFINE_float("explain_reg_weight", 0.05, "Weight for explanability regularization")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("num_source", 2, "Number of Source image: #seq - 1")
flags.DEFINE_integer("num_scales", 4, "Currently fixed to 4")
flags.DEFINE_integer("max_steps", 120000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_boolean("training_decay", True, "Continue training from previous checkpoint")
FLAGS = flags.FLAGS

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
        
    sfm = SfMLearner()
    sfm.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
