import numpy as np
import tensorflow as tf

import math
import os
import glob
import scipy.io



def save_txt_file(pred, name, SAVE_DIR):
    with open(os.path.join(SAVE_DIR, "{0}.txt".format(name)), 'w') as fp:
        for i in pred:
            # print(tuple(point.tolist()))
            fp.write("{0}\n".format(i))

def transform_tensor_to_image (tensor):
    t = tf.transpose(tensor, [0 , 2, 1, 3])
    return t[:,::-1, :, :]

def transform_voxel_to_match_image(tensor):
    tensor = tf.transpose(tensor, [0, 2, 1, 3, 4])
    tensor = tensor[:, ::-1, :, :, :]
    return tensor

def transform_image_to_match_voxel(tensor):
    tensor = tf.transpose(tensor, [0, 2, 1, 3])
    tensor = tensor[:, ::-1, :, :]
    return tensor

def np_transform_tensor_to_image (tensor):
    t = np.transpose(tensor, [0, 2, 1, 3])
    return t

