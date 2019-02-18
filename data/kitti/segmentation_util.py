from __future__ import division
import tensorflow as tf
import numpy as np
import scipy.misc

class segmentation_util(object):
    def __init__(self, seg_model_dir, img_height, img_width):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.GFile(seg_model_dir, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
            self.graph = graph

        self.img_input = self.graph.get_tensor_by_name('prefix/ImageTensor:0')
        self.seg_output = self.graph.get_tensor_by_name('prefix/SemanticPredictions:0')
        self.sess = tf.Session(graph = self.graph)
        self.img_height = img_height
        self.img_width = img_width

    def inference(self, img):
        #with tf.Session(graph=self.graph) as sess:
        with tf.device('/device:GPU:0'):
            img = np.expand_dims(img, axis = 0)
            probs = self.sess.run(self.seg_output, {self.img_input: img})
            img = np.squeeze(probs)
            mask = scipy.misc.imresize(img, (self.img_height, self.img_width), interp = "nearest")
        return mask
