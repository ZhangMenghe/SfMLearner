import os
import sys
import scipy
import numpy as np
import cv2
import tensorflow as tf
from helper import logits2image
import os
os.environ['TF_CPP_MIN_LOG_LEVLE'] = '2'

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("graph_dir", "", "Graph path")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
FLAGS = flags.FLAGS

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

graph = load_graph(FLAGS.graph_dir)
image_dir = FLAGS.dataset_dir

# DeepLabv3+ input and output tensors
image_input = graph.get_tensor_by_name('prefix/ImageTensor:0')
softmax = graph.get_tensor_by_name('prefix/SemanticPredictions:0')

# Create output directories in the image folder
if not os.path.exists(image_dir+'/segmented_images/'):
    os.mkdir(image_dir+'/segmented_images/')
if not os.path.exists(image_dir+'/segmented_images_colored/'):
    os.mkdir(image_dir+'/segmented_images_colored/') 

image_dir_segmented = image_dir+'segmented_images/'
image_dir_segmented_colored = image_dir+'segmented_images_colored/'

with tf.Session(graph=graph) as sess:
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".png"):
            img = scipy.misc.imread(os.path.join(image_dir, fname)) 
            img = np.expand_dims(img, axis=0)
            probs = sess.run(softmax, {image_input: img})
            img = tf.squeeze(probs).eval()
            print(img.shape)
            img_colored = logits2image(img)
            print(img_colored.shape)
            # print(os.path.join(image_dir_segmented+fname))
            cv2.imwrite(os.path.join(image_dir_segmented+fname),img)
            cv2.imwrite(os.path.join(image_dir_segmented_colored+fname),cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB))   
            # print(fname)
