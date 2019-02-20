import os
import sys
import scipy
import numpy as np
import imageio
import tensorflow as tf

from cityscapelabels import trainId2RGB

import os
os.environ['TF_CPP_MIN_LOG_LEVLE'] = '2'

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def logits2image(logits):
    logits = logits.astype(np.uint8)
    image = np.empty([logits.shape[0],logits.shape[1],3],dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j,:] = trainId2RGB[logits[i,j]]
    image = image.astype(np.uint8)
    return image

graph = load_graph("model/frozen_inference_graph.pb")#load_graph(sys.argv[1])
image_dir = "sample-images/"#sys.argv[2]

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
            img = imageio.imread(os.path.join(image_dir, fname)) 
            img = np.expand_dims(img, axis=0)
            probs = sess.run(softmax, {image_input: img})
            img = tf.squeeze(probs).eval()
            print(img.shape)
            img_colored = logits2image(img)
            print(img_colored.shape)
            # print(os.path.join(image_dir_segmented+fname))
            imageio.imwrite(os.path.join(image_dir_segmented+fname), img.astype(np.uint8))
            imageio.imwrite(os.path.join(image_dir_segmented_colored+fname), img_colored.astype(np.uint8))  
            # print(fname)
