from __future__ import division
import tensorflow as tf
import numpy as np
import scipy.misc

class segmentation_util(object):
    def __init__(self, seg_model_dir):
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

    def inference(self, img, img_height, img_weight):
        with tf.Session(graph=self.graph) as sess:
            img = np.expand_dims(img, axis = 0)
            probs = sess.run(self.seg_output, {self.img_input: img})
            img = tf.squeeze(probs).eval()
            mask = scipy.misc.imresize((img == 255).astype(int), (img_height, img_weight), interp = "nearest")
            for i in range(19):
                m = (img == i).astype(int)
                m = scipy.misc.imresize(m, (img_height, img_weight), interp = "nearest")
                mask = np.dstack((mask, m))
        return mask
