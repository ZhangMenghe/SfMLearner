import numpy as np
from glob import glob
import PIL.Image as pil
from os.path import isdir, exists

from scipy.misc import imresize
import tensorflow as tf

class SegmentHelper(object):
    def __init__(self, graph_path="/home/menghe/Github/DPNLearner/kitti-deeplab/model/frozen_inference_graph.pb", img_height=128, img_width=416):
        self.graph = self.load_graph(graph_path)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(graph = self.graph, config = self.config)
        self.image_input = self.graph.get_tensor_by_name('prefix/ImageTensor:0')
        softmax = self.graph.get_tensor_by_name('prefix/SemanticPredictions:0')
        self.seg_img_tensor = tf.squeeze(softmax)
        self.img_height = img_height
        self.img_width = img_width
        # self.h = hpy()
    
    def load_graph(self, frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
        return graph
    # @profile(precision=4)
    def apply(self, img):
        img = np.expand_dims(img, axis=0)
        segmented_img = self.sess.run(self.seg_img_tensor, {self.image_input: img})
        aft_resize = imresize(segmented_img,(self.img_height, self.img_width,3))
        return aft_resize
        
    def reset_session(self):
        # tf.reset_default_graph()
        self.sess.close()
        self.sess = tf.InteractiveSession(graph = self.graph, config = self.config)



# root_dir = '/home/menghe/Github/KITTI-FORMAT/'
# curr_dir= root_dir+ '2011_09_26_drive_0104_sync_02/'
dirs = glob('/home/menghe/Github/KITTI-FORMAT/*')
seg_loader = SegmentHelper()
dir_all = len(dirs)
for (n,curr_dir) in enumerate(dirs):
    if(isdir(curr_dir) == False):
        continue
    
    jpgs = glob(curr_dir + '/*.jpg')
    # folder = jpgs[0].split('/')[-2]
    for i, jpg in enumerate(jpgs):
        png_name = curr_dir + '/' + jpg.split('/')[-1].split('.')[0] + '-segque.png'
        if(exists(png_name)):
            continue
        with pil.open(jpg, 'r') as jpgimg:
            curr_img = np.array(jpgimg)
            src_img1= curr_img[:,:416,:]
            tgt_img = curr_img[:,416:832,:]
            src_img2 = curr_img[:,832:,:]
        
        segmentation1 = seg_loader.apply(src_img1)
        segmentation2 = seg_loader.apply(tgt_img)
        segmentation3 = seg_loader.apply(src_img2)
        segmentation = np.concatenate((segmentation1, segmentation2, segmentation3), axis=1)
        out_img = pil.fromarray(segmentation.astype(np.uint8))
        out_img.save(png_name)
    print('['+str(n)+ '/'+str(dir_all)+']  ' + "done")
