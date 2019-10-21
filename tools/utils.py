"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import tarfile
import zlib
import io
from PIL import Image
import random
import pprint
import scipy.misc
import numpy as np

from tools.rotation_utils import *


import tensorflow as tf
import tensorflow.contrib.slim as slim
import glob
import os
import random
import scipy.misc

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True):
  image = load_webp(image_path)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def load_webp(img_path):
    im = Image.open(img_path)
    return np.asarray(im)


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width,
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  if len(cropped_image.shape) != 3: #In case of binary mask with no channels:
    cropped_image = np.expand_dims(cropped_image, -1)
  return np.array(cropped_image)[:, :, :3]/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))

  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w

def to_bool(value):
    """
       Converts 'something' to boolean. Raises exception for invalid formats
           Possible True  values: 1, True, "1", "TRue", "yes", "y", "t"
           Possible False values: 0, False, None, [], {}, "", "0", "faLse", "no", "n", "f", 0.0, ...
    """
    if str(value).lower() == "true": return True
    if str(value).lower() == "false": return False
    raise Exception('Invalid value for boolean conversion: ' + str(value))

