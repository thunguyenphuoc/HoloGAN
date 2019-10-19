from __future__ import division
import os
import sys
from glob import glob
import json

code_path    = os.environ['TEXTURENET_CODE']
data_path    = os.environ['TEXTURENET_DATA']
results_path = os.environ['TEXTURENET_RESULTS']

import shutil
import tensorflow.contrib.slim as slim
import numpy as np
import random
import tensorflow as tf
import math

with open(sys.argv[1], 'r') as fh:
    cfg=json.load(fh)
IMAGE_PATH       = os.path.join(data_path, cfg['image_path'])
IMAGE_NAME_PATH = os.path.join(data_path, cfg['image_name_path'])
SAMPLE_SAVE = os.path.join(results_path, cfg['sample_save'])
MODEL_SAVE  = os.path.join(SAMPLE_SAVE, cfg['trained_model_name'])
LOGDIR = os.path.join(SAMPLE_SAVE, "log")


from tools.ops import *
from tools.data_utils import image_data_loader, horizontal_flip
from tools.utils import get_image, get_image_mvc, merge, inverse_transform, to_bool
from tools.rotation_utils import *
from tools.model_utils import transform_voxel_to_match_image



def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
#----------------------------------------------------------------------------

class HoloGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='lsun',
         input_fname_pattern='*.webp', checkpoint_dir=None, generator3D=True, with_rotation=True, sample_dir=SAMPLE_SAVE):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim
    self.c_dim = c_dim


    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))


  def build(self, build_func_name):
      build_func = eval("self." + build_func_name)
      build_func()

  def sampling_Z(self, z_dim, type="uniform"):
      if str.lower(type) == "uniform":
          return np.random.uniform(-1., 1., (cfg['batch_size'], z_dim))
      else:
          return np.random.normal(0, 1, (cfg['batch_size'], z_dim))

  def AdaIn(self, features, scale, bias):
      """
      Adaptive instance normalization component. Works with both 4D and 5D tensors
      :features: features to be normalized
      :scale: scaling factor. This would otherwise be calculated as the sigma from a "style" features in style transfer
      :bias: bias factor. This would otherwise be calculated as the mean from a "style" features in style transfer
      """
      mean, variance = tf.nn.moments(features, list(range(len(features.get_shape())))[1:-1], keep_dims=True) #Only consider spatial dimension
      sigma = tf.rsqrt(variance + 1e-8)
      normalized = (features - mean) * sigma
      scale_broadcast = tf.reshape(scale, tf.shape(mean))
      bias_broadcast = tf.reshape(bias, tf.shape(mean))
      normalized = scale_broadcast * normalized
      normalized += bias_broadcast
      return normalized

  def linear_classifier(self, features, scope = "lin_class", stddev=0.02, reuse=False):
      with tf.variable_scope(scope) as sc:
          w = tf.get_variable('w', [features.get_shape()[-1], 1],
                              initializer=tf.random_normal_initializer(stddev=stddev))
          b = tf.get_variable('biases', 1, initializer=tf.constant_initializer(0.0))
          logits = tf.matmul(features, w) + b
          return   tf.nn.sigmoid(logits), logits

  def instance_norm(self, input, name="instance_norm", return_mean=False):
      """
      Taken from https://github.com/xhujoy/CycleGAN-tensorflow/blob/master/module.py
      :param input:
      :param name:
      :return:
      """
      with tf.variable_scope(name):
          depth = input.get_shape()[3]
          scale = tf.get_variable("scale", [depth],
                                  initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
          offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
          mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
          epsilon = 1e-5
          inv = tf.rsqrt(variance + epsilon)
          normalized = (input - mean) * inv
          if return_mean:
              return scale * normalized + offset, mean, variance
          else:
              return scale * normalized + offset

  def build_model(self):
    self.view_in = tf.placeholder(tf.float32, [None, 6], name='view_in')
    self.inputs = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, self.c_dim], name='real_images')
    self.z = tf.placeholder(tf.float32, [None, cfg['z_dim']], name='z')
    self.is_training = tf.placeholder(tf.bool, [], name='is_training')
    # self.z_sum = histogram_summary("z", self.z)

    inputs = self.inputs

    gen_func = eval("self." + (cfg['generator']))
    dis_func = eval("self." + (cfg['discriminator']))
    self.gen_view_func = eval(cfg['view_func'])

    if str.lower(str(cfg["z_to_w"])) == "true":
        w = self.z_to_w(self.z, cfg['z_dim'])
        self.G = gen_func(w, self.view_in)
    else:
        self.G = gen_func(self.z, self.view_in)

    if str.lower(str(cfg["style_disc"])) == "true":
        print("Style Disc")
        self.D, self.D_logits, _, self.d_h1_r, self.d_h2_r, self.d_h3_r, self.d_h4_r = dis_func(inputs, cont_dim=cfg['z_dim'], reuse=False)
        self.D_, self.D_logits_, self.Q_c_given_x, self.d_h1_f, self.d_h2_f, self.d_h3_f, self.d_h4_f = dis_func(self.G, cont_dim=cfg['z_dim'], reuse=True)
        if str.lower(str(cfg["Softmax_loss"])) == "true":
            print("Style_softmax")
            self.d_h1_loss = tf.reduce_mean(tf.nn.softplus(self.d_h1_f)) \
                             + tf.reduce_mean(tf.nn.softplus(-self.d_h1_r))
            self.d_h2_loss = tf.reduce_mean(tf.nn.softplus(self.d_h2_f)) \
                             + tf.reduce_mean(tf.nn.softplus(-self.d_h2_r))
            self.d_h3_loss = tf.reduce_mean(tf.nn.softplus(self.d_h3_f)) \
                             + tf.reduce_mean(tf.nn.softplus(-self.d_h3_r))
            self.d_h4_loss = tf.reduce_mean(tf.nn.softplus(self.d_h4_f)) \
                             + tf.reduce_mean(tf.nn.softplus(-self.d_h4_r))
        elif str.lower(str(cfg["StyleD_label_flip_correct"])) == "true":
            print("New style D")
            self.d_h1_loss = cfg["DStyle_lambda"] * (tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.ones_like(self.d_h1_r)))\
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_f, tf.zeros_like(self.d_h1_f))))
            self.d_h2_loss = cfg["DStyle_lambda"] * (tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.ones_like(self.d_h2_r))) \
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_f, tf.zeros_like(self.d_h2_f))))
            self.d_h3_loss = cfg["DStyle_lambda"] * (tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.ones_like(self.d_h3_r))) \
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_f, tf.zeros_like(self.d_h3_f))))
            self.d_h4_loss = cfg["DStyle_lambda"] * (tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_r, tf.ones_like(self.d_h4_r))) \
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_f, tf.zeros_like(self.d_h4_f))))
        else:
            #===============================================================================================================
            #This seems to be the wrong implementation at tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.zeros_like(self.d_h1_f)))
            #However, this method was used in the best model so far: 250\190217_AdaIN_carsCroppedAugmented_lambda_1.0_d_z200_eta0.00005_lrelu_smallconvB3foreProj_res128_decayEta
            #In these cases, cfg["DStyle_lambda"]  = 1.0
            print("Old style D")
            self.d_h1_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.ones_like(self.d_h1_r)))\
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.zeros_like(self.d_h1_f)))
            self.d_h2_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.ones_like(self.d_h2_r))) \
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.zeros_like(self.d_h2_f)))
            self.d_h3_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.ones_like(self.d_h3_r))) \
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.zeros_like(self.d_h3_f)))
            self.d_h4_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_r, tf.ones_like(self.d_h4_r))) \
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_r, tf.zeros_like(self.d_h4_f)))
            #===============================================================================================================
    else:
        self.D, self.D_logits, _ = dis_func(inputs, cont_dim=cfg['z_dim'], reuse=False)
        self.D_, self.D_logits_, self.Q_c_given_x = dis_func(self.G, cont_dim=cfg['z_dim'], reuse=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    # self.G_sum = image_summary("G", self.G)

    if str.lower(str(cfg["least_square_GAN"])) == "true":
        print("Least square GAN")
        self.d_loss_real = tf.reduce_mean(tf.squared_difference(self.D, 1))
        self.d_loss_fake = tf.reduce_mean(tf.square(self.D_))
        self.g_loss = tf.reduce_mean(tf.squared_difference(self.D_, 1))
        self.d_loss = 0.5 * (self.d_loss_real + self.d_loss_fake)

    elif str.lower(str(cfg["hinge_GAN"])) == "true":
        print("Hinge Loss GAN")
        self.d_loss_real = tf.reduce_mean(tf.nn.relu(1. - self.D_logits))
        self.d_loss_fake = tf.reduce_mean(tf.nn.relu(1. + self.D_logits_))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = - tf.reduce_mean(self.D_logits_)

    elif str.lower(str(cfg["RA_GAN"])) == "true":
        print("Relativistic GAN")
        """
        (BCE_stable(y_pred - torch.mean(y_pred_fake), y) + BCE_stable(y_pred_fake - torch.mean(y_pred), y2))/2
        errG = ((BCE_stable(y_pred - torch.mean(y_pred_fake), y2) + BCE_stable(y_pred_fake - torch.mean(y_pred), y))/2
        """
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits - tf.reduce_mean(self.D_logits_), tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_ - tf.reduce_mean(self.D_logits), tf.zeros_like(self.D)))
        self.d_loss = (self.d_loss_real + self.d_loss_fake) * 0.5
        self.g_loss = (tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits - tf.reduce_mean(self.D_logits_), tf.zeros_like(self.D)))
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_ - tf.reduce_mean(self.D_logits), tf.ones_like(self.D)))) * 0.5

    elif str.lower(str(cfg["RA_hinge_GAN"])) == "true":
        print("Relativistic Hinge GAN")
        """
        (BCE_stable(y_pred - torch.mean(y_pred_fake), y) + BCE_stable(y_pred_fake - torch.mean(y_pred), y2))/2
        errG = ((BCE_stable(y_pred - torch.mean(y_pred_fake), y2) + BCE_stable(y_pred_fake - torch.mean(y_pred), y))/2
        """
        self.d_loss_real = tf.reduce_mean(tf.nn.relu(1. - (self.D_logits - tf.reduce_mean(self.D_logits_))))
        self.d_loss_fake = tf.reduce_mean(tf.nn.relu(1. + (self.D_logits_ - tf.reduce_mean(self.D_logits))))
        self.d_loss = (self.d_loss_real + self.d_loss_fake) * 0.5

        self.g_loss_real = tf.reduce_mean(tf.nn.relu(1. + (self.D_logits - tf.reduce_mean(self.D_logits_))))
        self.g_loss_fake = tf.reduce_mean(tf.nn.relu(1. - (self.D_logits_ - tf.reduce_mean(self.D_logits))))
        self.g_loss =  (self.g_loss_real + self.g_loss_fake) * 0.5
    elif str.lower(str(cfg["Softmax_loss"])) == "true":
        print("Softmax loss")
        self.g_loss = tf.reduce_mean(tf.nn.softplus(-self.D_logits_))
        self.d_loss_real = tf.reduce_mean(tf.nn.softplus(self.D_logits_))
        self.d_loss_fake = tf.reduce_mean(tf.nn.softplus(-self.D_logits))
        self.d_loss = self.d_loss_real + self.d_loss_fake
    else:
        print("Normal GAN")
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        if str.lower(str(cfg["change_G_loss"])) == "true":
            self.g_loss = tf.reduce_mean(-tf.log(tf.sigmoid(self.D_logits_) + 1e-12))
        else:
            self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))


    if str.lower(str(cfg["style_disc"])) == "true":
        print("Style disc")
        self.d_loss = self.d_loss + self.d_h1_loss + self.d_h2_loss + self.d_h3_loss + self.d_h4_loss
    #====================================================================================================================
    #INFOGAN loss

    if str.lower(str(cfg["predict_view"])) == "true":
        self.q_loss = cfg["lambda_latent"] * tf.reduce_mean(tf.square(self.Q_c_given_x - self.view_in))
    else:
        self.q_loss = cfg["lambda_latent"] * tf.reduce_mean(tf.square(self.Q_c_given_x - self.z))
    # self.q_loss = 0
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

    self.d_loss = self.d_loss + self.q_loss
    self.g_loss = self.g_loss + self.q_loss

    # =====================================================================================================================
    #Temporal coherence loss
    if str.lower(str(cfg["add_view_noise"])) == "true":
        print("WITH G NOISE")
        # Add pertubation
        rotation_noise = tf.random_normal(shape=(tf.shape(self.view_in)[0], 2), mean=0.0, stddev=1, dtype=tf.float32) * math.pi / 180.0
        zeros = tf.zeros(shape=(tf.shape(self.view_in)[0], 1))
        rotation_noise = tf.concat([rotation_noise, zeros, zeros, zeros, zeros], axis=1)  # Concatenate with scale and translation dimension
        self.view_in_noisy = self.view_in + rotation_noise
        self.G_noisy = gen_func(self.z, self.view_in_noisy, reuse=True)

        # G regularisers
        self.g_regulariser = tf.losses.mean_squared_error(labels=self.G, predictions=self.G_noisy)
        self.g_loss += cfg['g_regulariser'] * self.g_regulariser
    # =====================================================================================================================
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def build_model_2(self):
    self.view_in = tf.placeholder(tf.float32, [None, 6], name='view_in')
    self.inputs = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, self.c_dim], name='real_images')
    self.z = tf.placeholder(tf.float32, [None, cfg['z_dim']], name='z1')
    self.z2 = tf.placeholder(tf.float32, [None, cfg['z_dim']], name='z2')
    self.is_training = tf.placeholder(tf.bool, [], name='is_training')

    inputs = self.inputs

    gen_func = eval("self." + (cfg['generator']))
    dis_func = eval("self." + (cfg['discriminator']))

    self.G = gen_func(self.z, self.z2, self.view_in)

    if str.lower(str(cfg["style_disc"])) == "true":
        print("Style Disc")
        self.D, self.D_logits, _, self.d_h1_r, self.d_h2_r, self.d_h3_r, self.d_h4_r = dis_func(inputs, cont_dim=cfg['z_dim'], reuse=False)
        self.D_, self.D_logits_, self.Q_c_given_x, self.d_h1_f, self.d_h2_f, self.d_h3_f, self.d_h4_f = dis_func(self.G, cont_dim=cfg['z_dim'], reuse=True)
        if str.lower(str(cfg["Softmax_loss"])) == "true":
            print("Style_softmax")
            self.d_h1_loss = tf.reduce_mean(tf.nn.softplus(self.d_h1_f)) \
                             + tf.reduce_mean(tf.nn.softplus(-self.d_h1_r))
            self.d_h2_loss = tf.reduce_mean(tf.nn.softplus(self.d_h2_f)) \
                             + tf.reduce_mean(tf.nn.softplus(-self.d_h2_r))
            self.d_h3_loss = tf.reduce_mean(tf.nn.softplus(self.d_h3_f)) \
                             + tf.reduce_mean(tf.nn.softplus(-self.d_h3_r))
            self.d_h4_loss = tf.reduce_mean(tf.nn.softplus(self.d_h4_f)) \
                             + tf.reduce_mean(tf.nn.softplus(-self.d_h4_r))
        else:
            self.d_h1_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.ones_like(self.d_h1_r)))\
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.zeros_like(self.d_h1_f)))
            self.d_h2_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.ones_like(self.d_h2_r))) \
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.zeros_like(self.d_h2_f)))
            self.d_h3_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.ones_like(self.d_h3_r))) \
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.zeros_like(self.d_h3_f)))
            self.d_h4_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_r, tf.ones_like(self.d_h4_r))) \
                             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_r, tf.zeros_like(self.d_h4_f)))
    else:
        self.D, self.D_logits, _ = dis_func(inputs, cont_dim=cfg['z_dim'], reuse=False)
        self.D_, self.D_logits_, self.Q_c_given_x = dis_func(self.G, cont_dim=cfg['z_dim'], reuse=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    # self.G_sum = image_summary("G", self.G)

    if str.lower(str(cfg["least_square_GAN"])) == "true":
        print("Least square GAN")
        self.d_loss_real = tf.reduce_mean(tf.squared_difference(self.D, 1))
        self.d_loss_fake = tf.reduce_mean(tf.square(self.D_))
        self.g_loss = tf.reduce_mean(tf.squared_difference(self.D_, 1))
        self.d_loss = 0.5 * (self.d_loss_real + self.d_loss_fake)

    elif str.lower(str(cfg["hinge_GAN"])) == "true":
        print("Hinge Loss GAN")
        self.d_loss_real = tf.reduce_mean(tf.nn.relu(1. - self.D_logits))
        self.d_loss_fake = tf.reduce_mean(tf.nn.relu(1. + self.D_logits_))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = - tf.reduce_mean(self.D_logits_)

    elif str.lower(str(cfg["RA_GAN"])) == "true":
        print("Relativistic GAN")
        """
        (BCE_stable(y_pred - torch.mean(y_pred_fake), y) + BCE_stable(y_pred_fake - torch.mean(y_pred), y2))/2
        errG = ((BCE_stable(y_pred - torch.mean(y_pred_fake), y2) + BCE_stable(y_pred_fake - torch.mean(y_pred), y))/2
        """
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits - tf.reduce_mean(self.D_logits_), tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_ - tf.reduce_mean(self.D_logits), tf.zeros_like(self.D)))
        self.d_loss = (self.d_loss_real + self.d_loss_fake) * 0.5
        self.g_loss = (tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits - tf.reduce_mean(self.D_logits_), tf.zeros_like(self.D)))
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_ - tf.reduce_mean(self.D_logits), tf.ones_like(self.D)))) * 0.5

    elif str.lower(str(cfg["RA_hinge_GAN"])) == "true":
        print("Relativistic Hinge GAN")
        """
        (BCE_stable(y_pred - torch.mean(y_pred_fake), y) + BCE_stable(y_pred_fake - torch.mean(y_pred), y2))/2
        errG = ((BCE_stable(y_pred - torch.mean(y_pred_fake), y2) + BCE_stable(y_pred_fake - torch.mean(y_pred), y))/2
        """
        self.d_loss_real = tf.reduce_mean(tf.nn.relu(1. - (self.D_logits - tf.reduce_mean(self.D_logits_))))
        self.d_loss_fake = tf.reduce_mean(tf.nn.relu(1. + (self.D_logits_ - tf.reduce_mean(self.D_logits))))
        self.d_loss = (self.d_loss_real + self.d_loss_fake) * 0.5

        self.g_loss_real = tf.reduce_mean(tf.nn.relu(1. + (self.D_logits - tf.reduce_mean(self.D_logits_))))
        self.g_loss_fake = tf.reduce_mean(tf.nn.relu(1. - (self.D_logits_ - tf.reduce_mean(self.D_logits))))
        self.g_loss =  (self.g_loss_real + self.g_loss_fake) * 0.5
    elif str.lower(str(cfg["Softmax_loss"])) == "true":
        print("Softmax loss")
        self.g_loss = tf.reduce_mean(tf.nn.softplus(-self.D_logits_))
        self.d_loss_real = tf.reduce_mean(tf.nn.softplus(self.D_logits_))
        self.d_loss_fake = tf.reduce_mean(tf.nn.softplus(-self.D_logits))
        self.d_loss = self.d_loss_real + self.d_loss_fake
    else:
        print("Normal GAN")
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        if str.lower(str(cfg["change_G_loss"])) == "true":
            self.g_loss = tf.reduce_mean(-tf.log(tf.sigmoid(self.D_logits_) + 1e-12))
        else:
            self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))


    if str.lower(str(cfg["style_disc"])) == "true":
        print("Style disc")
        self.d_loss = self.d_loss + self.d_h1_loss + self.d_h2_loss + self.d_h3_loss + self.d_h4_loss
    #====================================================================================================================
    #INFOGAN loss

    if str.lower(str(cfg["predict_view"])) == "true":
        self.q_loss = cfg["lambda_latent"] * tf.reduce_mean(tf.square(self.Q_c_given_x - self.view_in))
    else:
        self.q_loss = cfg["lambda_latent"] * tf.reduce_mean(tf.square(self.Q_c_given_x - self.z))
    # self.q_loss = 0
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

    self.d_loss = self.d_loss + self.q_loss
    self.g_loss = self.g_loss + self.q_loss

    # =====================================================================================================================
    #Temporal coherence loss
    if str.lower(str(cfg["add_view_noise"])) == "true":
        print("WITH G NOISE")
        # Add pertubation
        rotation_noise = tf.random_normal(shape=(tf.shape(self.view_in)[0], 2), mean=0.0, stddev=1, dtype=tf.float32) * math.pi / 180.0
        zeros = tf.zeros(shape=(tf.shape(self.view_in)[0], 1))
        rotation_noise = tf.concat([rotation_noise, zeros, zeros, zeros, zeros], axis=1)  # Concatenate with scale and translation dimension
        self.view_in_noisy = self.view_in + rotation_noise
        self.G_noisy = gen_func(self.z, self.view_in_noisy, reuse=True)

        # G regularisers
        self.g_regulariser = tf.losses.mean_squared_error(labels=self.G, predictions=self.G_noisy)
        self.g_loss += cfg['g_regulariser'] * self.g_regulariser
    # =====================================================================================================================
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def z_mapping_function(self, z, output_channel, scope='z_mapping', act="relu", stddev=0.02):
      with tf.variable_scope(scope) as sc:
          w = tf.get_variable('w', [z.get_shape()[-1], output_channel * 2],
                              initializer=tf.random_normal_initializer(stddev=stddev))
          b = tf.get_variable('biases', output_channel * 2, initializer=tf.constant_initializer(0.0))
          if act == "relu":
              print("Relu")
              out = tf.nn.relu(tf.matmul(z, w) + b)
          elif act == "prelu":
              print("prelu")
              out = prelu(tf.matmul(z, w) + b)
          else:
              print('lrelu')
              out = lrelu(tf.matmul(z, w) + b)
          return out[:, :output_channel], out[:, output_channel:]

  def train(self, config):
      if str.lower(str(cfg["adam_D"])) == "true":
          print("Adam")
          d_optim = tf.train.AdamOptimizer(cfg['d_eta'], beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.d_loss, var_list=self.d_vars)
      else:
          print("SGD")
          d_optim = tf.train.MomentumOptimizer(cfg['d_eta'], -0.3).minimize(self.d_loss, var_list=self.d_vars)

      if str.lower(str(cfg["adam_G"])) == "true":
          print("Adam")
          g_optim = tf.train.AdamOptimizer(cfg['g_eta'], beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.g_loss, var_list=self.g_vars)
      else:
          print("SGD")
          g_optim = tf.train.MomentumOptimizer(cfg['g_eta'], -0.3).minimize(self.g_loss, var_list=self.g_vars)

      tf.global_variables_initializer().run()

      shutil.copyfile(sys.argv[1], os.path.join(LOGDIR, 'config.json'))
      self.g_sum = merge_summary([self.d__sum,
                                  self.d_loss_fake_sum, self.g_loss_sum])
      self.d_sum = merge_summary( [self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
      self.writer = SummaryWriter(os.path.join(SAMPLE_SAVE, "log"), self.sess.graph)

      # Sample one fixed Z and view parameters to test the sampling

      sample_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
      if "_3" in str(cfg['view_func']) or "_4" in str(cfg['view_func']):
          sample_azimuth = np.random.uniform(0, 359, cfg['batch_size'])
          sample_view = self.gen_view_func(cfg['batch_size'], sample_azimuth, cfg['ele_low'], cfg['ele_high'],
                                           cfg['scale_low'], cfg['scale_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=to_bool(str(cfg['with_translation'])),
                                           with_scale=to_bool(str(cfg['with_translation'])))
      elif "_5" in str(cfg['view_func']) in str(cfg['view_func']):
          sample_azimuth = np.random.uniform(0, 359, cfg['batch_size'])
          sample_scale = np.random.uniform(cfg['scale_low'], cfg['scale_high'], cfg['batch_size'])
          sample_view = self.gen_view_func(cfg['batch_size'], sample_azimuth, sample_scale,
                                           cfg['ele_low'], cfg['ele_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=to_bool(str(cfg['with_translation'])),
                                           with_scale=to_bool(str(cfg['with_translation'])))
      else:
          sample_view =  self.gen_view_func(cfg['batch_size'], cfg['ele_low'], cfg['ele_high'],
                                                             cfg['azi_low'], cfg['azi_high'],
                                                             cfg['scale_low'], cfg['scale_high'],
                                                             cfg['x_low'], cfg['x_high'],
                                                             cfg['y_low'], cfg['y_high'],
                                                             cfg['z_low'], cfg['z_high'],
                                                             with_translation = to_bool(str(cfg['with_translation'])),
                                                             with_scale = to_bool(str(cfg['with_translation'])))

      sample_files = self.data[0:cfg['batch_size']]

      if config.dataset == "shoes" or config.dataset == "mvc" or config.dataset == "cats" or config.dataset == "cars":
          sample_images = [get_image_mvc(sample_file, self.output_height) for sample_file in sample_files]

      else:
          sample_images = [get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for sample_file in sample_files]
      sample_inputs = np.array(sample_images).astype(np.float32)

      counter = 1
      start_time = time.time()
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")

      self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))
      for epoch in range(cfg['max_epochs']):
          if config.dataset == "cars":
              print("MAKE SCALE AND AZIMUTH")
              self.azimuth_all, self.scale_all = create_car_azimuth_scale(cfg['scale_low'], cfg['scale_high'])

          random.shuffle(self.data)
          batch_idxs = min(len(self.data), config.train_size) // cfg['batch_size']

          for idx in range(0, batch_idxs):
              batch_files = self.data[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
              if config.dataset == "shoes" or config.dataset == "mvc" or config.dataset == "cats" or config.dataset == "cars":
                  batch_images = [get_image_mvc(batch_file, self.output_height) for batch_file in batch_files]
              else:
                  batch_images = [get_image(batch_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for batch_file in batch_files]


              #Randomly flip images
              if config.dataset is not "cars" or config.dataset is not "carsOld":
                  if np.random.uniform(0.1, 1.0) <= 0.5:
                      try:
                          batch_images = horizontal_flip(batch_images)
                      except:
                          print("cannot flip")
              batch_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
              if "_3" in str(cfg['view_func']) or "_4" in str(cfg['view_func']):
                  batch_azimuth = self.azimuth_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_view = self.gen_view_func(cfg['batch_size'], batch_azimuth, cfg['ele_low'], cfg['ele_high'],
                                                  cfg['scale_low'], cfg['scale_high'],
                                                  cfg['x_low'], cfg['x_high'],
                                                  cfg['y_low'], cfg['y_high'],
                                                  cfg['z_low'], cfg['z_high'],
                                                  with_translation=to_bool(str(cfg['with_translation'])),
                                                  with_scale=to_bool(str(cfg['with_translation'])))
              elif "_5" in str(cfg['view_func']) in str(cfg['view_func']):
                  batch_azimuth = self.azimuth_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_scale = self.scale_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_view = self.gen_view_func(cfg['batch_size'], batch_azimuth, batch_scale,
                                                   cfg['ele_low'], cfg['ele_high'],
                                                   cfg['x_low'], cfg['x_high'],
                                                   cfg['y_low'], cfg['y_high'],
                                                   cfg['z_low'], cfg['z_high'],
                                                   with_translation=to_bool(str(cfg['with_translation'])),
                                                   with_scale=to_bool(str(cfg['with_translation'])))
              else:
                  batch_view = self.gen_view_func(cfg['batch_size'], cfg['ele_low'], cfg['ele_high'],
                                                  cfg['azi_low'], cfg['azi_high'],
                                                  cfg['scale_low'], cfg['scale_high'],
                                                  cfg['x_low'], cfg['x_high'],
                                                  cfg['y_low'], cfg['y_high'],
                                                  cfg['z_low'], cfg['z_high'],
                                                  with_translation=to_bool(str(cfg['with_translation'])),
                                                  with_scale=to_bool(str(cfg['with_translation'])))

              feed = {self.inputs: batch_images,
                      self.z: batch_z,
                      self.view_in: batch_view,
                      self.is_training: True}
              # try:
              # Update D network
              _, summary_str = self.sess.run([d_optim, self.d_sum],
                                             feed_dict=feed)
              self.writer.add_summary(summary_str, counter)
              # Update G network
              _, summary_str = self.sess.run([g_optim, self.g_sum],
                                             feed_dict=feed)
              self.writer.add_summary(summary_str, counter)

              if str.lower(str(cfg["g_twice"])) == "true":
                  print("Train G Twice")
                  # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                  _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                 feed_dict=feed)
                  self.writer.add_summary(summary_str, counter)

              errD_fake = self.d_loss_fake.eval(feed)
              errD_real = self.d_loss_real.eval(feed)
              errG = self.g_loss.eval(feed)
              errQ = self.q_loss.eval(feed)
              # except:
              #       print("FAIL")
              #       continue



              counter += 1
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                       time.time() - start_time, errD_fake + errD_real, errG, errQ))

              if np.mod(counter, 500) == 1:
                  if self.with_rotation:
                      feed_eval = {self.inputs: sample_inputs,
                                   self.z: sample_z,
                                   self.view_in: sample_view,
                                   self.is_training: False}


                  samples, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict=feed_eval)
                  ren_img = inverse_transform(samples)
                  ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
                  try:
                      scipy.misc.imsave(
                          os.path.join(SAMPLE_SAVE, "{0}_GAN.png".format(counter)),
                          merge(ren_img, [cfg['batch_size'] // 4, 4]))

                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                  except:
                      scipy.misc.imsave(
                          os.path.join(SAMPLE_SAVE, "{0}_GAN.png".format(counter)),
                          ren_img[0])
                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))


              if np.mod(counter, 500) == 2:
                  self.save(config.checkpoint_dir, counter)

  def train_2(self, config):
      self.d_lr_in = tf.placeholder(tf.float32, None, name='d_eta')
      self.g_lr_in = tf.placeholder(tf.float32, None, name='d_eta')

      if str.lower(str(cfg["adam_D"])) == "true":
          print("Adam")
          d_optim = tf.train.AdamOptimizer(self.d_lr_in, beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.d_loss, var_list=self.d_vars)
      else:
          print("SGD")
          d_optim = tf.train.MomentumOptimizer(self.d_lr_in, -0.3).minimize(self.d_loss, var_list=self.d_vars)

      if str.lower(str(cfg["adam_G"])) == "true":
          print("Adam")
          g_optim = tf.train.AdamOptimizer(self.g_lr_in, beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.g_loss, var_list=self.g_vars)
      else:
          print("SGD")
          g_optim = tf.train.MomentumOptimizer(self.g_lr_in, -0.3).minimize(self.g_loss, var_list=self.g_vars)

      tf.global_variables_initializer().run()

      shutil.copyfile(sys.argv[1], os.path.join(LOGDIR, 'config.json'))
      self.g_sum = merge_summary([self.d__sum,
                                  self.d_loss_fake_sum, self.g_loss_sum])
      self.d_sum = merge_summary( [self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
      self.writer = SummaryWriter(os.path.join(SAMPLE_SAVE, "log"), self.sess.graph)

      # Sample one fixed Z and view parameters to test the sampling
      sample_z = self.sampling_Z()
      if "_3" in str(cfg['view_func']) or "_4" in str(cfg['view_func']):
          sample_azimuth = np.random.uniform(0, 359, cfg['batch_size'])
          sample_view = self.gen_view_func(cfg['batch_size'], sample_azimuth, cfg['ele_low'], cfg['ele_high'],
                                           cfg['scale_low'], cfg['scale_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=to_bool(str(cfg['with_translation'])),
                                           with_scale=to_bool(str(cfg['with_translation'])))
      elif "_5" in str(cfg['view_func']) in str(cfg['view_func']):
          sample_azimuth = np.random.uniform(0, 359, cfg['batch_size'])
          sample_scale = np.random.uniform(cfg['scale_low'], cfg['scale_high'], cfg['batch_size'])
          sample_view = self.gen_view_func(cfg['batch_size'], sample_azimuth, sample_scale,
                                           cfg['ele_low'], cfg['ele_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=to_bool(str(cfg['with_translation'])),
                                           with_scale=to_bool(str(cfg['with_translation'])))
      else:
          sample_view =  self.gen_view_func(cfg['batch_size'], cfg['ele_low'], cfg['ele_high'],
                                                             cfg['azi_low'], cfg['azi_high'],
                                                             cfg['scale_low'], cfg['scale_high'],
                                                             cfg['x_low'], cfg['x_high'],
                                                             cfg['y_low'], cfg['y_high'],
                                                             cfg['z_low'], cfg['z_high'],
                                                             with_translation = to_bool(str(cfg['with_translation'])),
                                                             with_scale = to_bool(str(cfg['with_translation'])))

      sample_files = self.data[0:cfg['batch_size']]

      if config.dataset == "shoes" or config.dataset == "mvc" or config.dataset == "cats" or config.dataset == "cars":
          sample_images = [get_image_mvc(sample_file, self.output_height) for sample_file in sample_files]

      else:
          sample_images = [get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for sample_file in sample_files]
      sample_inputs = np.array(sample_images).astype(np.float32)

      counter = 1
      start_time = time.time()
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")

      self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))
      d_lr = cfg['d_eta']
      g_lr = cfg['g_eta']
      for epoch in range(cfg['max_epochs']):
          d_lr = d_lr if epoch < cfg['epoch_step'] else d_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])
          g_lr = g_lr if epoch < cfg['epoch_step'] else g_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])

          if config.dataset == "cars":
              self.azimuth_all, self.scale_all = create_car_azimuth_scale(cfg['scale_low'], cfg['scale_high'])
          random.shuffle(self.data)
          batch_idxs = min(len(self.data), config.train_size) // cfg['batch_size']

          for idx in range(0, batch_idxs):
              batch_files = self.data[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
              if config.dataset == "shoes" or config.dataset == "mvc" or config.dataset == "cats" or config.dataset == "cars":
                  batch_images = [get_image_mvc(batch_file, self.output_height) for batch_file in batch_files]
              else:
                  batch_images = [get_image(batch_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for batch_file in batch_files]


              #Randomly flip images
              if config.dataset is not "cars" or config.dataset is not "carsOld":
                  if np.random.uniform(0.1, 1.0) <= 0.5:
                      try:
                          batch_images = horizontal_flip(batch_images)
                      except:
                          print("cannot flip")
              batch_z = self.sampling_Z()
              if "_3" in str(cfg['view_func']) or "_4" in str(cfg['view_func']):
                  batch_azimuth = self.azimuth_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_view = self.gen_view_func(cfg['batch_size'], batch_azimuth, cfg['ele_low'], cfg['ele_high'],
                                                  cfg['scale_low'], cfg['scale_high'],
                                                  cfg['x_low'], cfg['x_high'],
                                                  cfg['y_low'], cfg['y_high'],
                                                  cfg['z_low'], cfg['z_high'],
                                                  with_translation=to_bool(str(cfg['with_translation'])),
                                                  with_scale=to_bool(str(cfg['with_translation'])))
              elif "_5" in str(cfg['view_func']) in str(cfg['view_func']):
                  batch_azimuth = self.azimuth_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_scale = self.scale_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_view = self.gen_view_func(cfg['batch_size'], batch_azimuth, batch_scale,
                                                   cfg['ele_low'], cfg['ele_high'],
                                                   cfg['x_low'], cfg['x_high'],
                                                   cfg['y_low'], cfg['y_high'],
                                                   cfg['z_low'], cfg['z_high'],
                                                   with_translation=to_bool(str(cfg['with_translation'])),
                                                   with_scale=to_bool(str(cfg['with_translation'])))
              else:
                  batch_view = self.gen_view_func(cfg['batch_size'], cfg['ele_low'], cfg['ele_high'],
                                                  cfg['azi_low'], cfg['azi_high'],
                                                  cfg['scale_low'], cfg['scale_high'],
                                                  cfg['x_low'], cfg['x_high'],
                                                  cfg['y_low'], cfg['y_high'],
                                                  cfg['z_low'], cfg['z_high'],
                                                  with_translation=to_bool(str(cfg['with_translation'])),
                                                  with_scale=to_bool(str(cfg['with_translation'])))

              feed = {self.inputs: batch_images,
                      self.z: batch_z,
                      self.view_in: batch_view,
                      self.d_lr_in: d_lr,
                      self.g_lr_in: g_lr,
                      self.is_training: True}
              try:
                  # Update D network
                  _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                 feed_dict=feed)
                  self.writer.add_summary(summary_str, counter)
                  # Update G network
                  _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                 feed_dict=feed)
                  self.writer.add_summary(summary_str, counter)
                  print("Learning rate " + str(d_lr))
                  if str.lower(str(cfg["g_twice"])) == "true":
                      print("Train G Twice")
                      # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                      _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                     feed_dict=feed)
                      self.writer.add_summary(summary_str, counter)

                  errD_fake = self.d_loss_fake.eval(feed)
                  errD_real = self.d_loss_real.eval(feed)
                  errG = self.g_loss.eval(feed)
                  errQ = self.q_loss.eval(feed)
              except:
                    print("FAIL")
                    continue



              counter += 1
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                       time.time() - start_time, errD_fake + errD_real, errG, errQ))

              if np.mod(counter, 500) == 1:
                  if self.with_rotation:
                      feed_eval = {self.inputs: sample_inputs,
                                   self.z: sample_z,
                                   self.view_in: sample_view,
                                   self.d_lr_in: d_lr,
                                   self.g_lr_in: g_lr,
                                   self.is_training: False}


                  samples, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict=feed_eval)
                  ren_img = inverse_transform(samples)
                  ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
                  try:
                      scipy.misc.imsave(
                          os.path.join(SAMPLE_SAVE, "{0}_GAN.png".format(counter)),
                          merge(ren_img, [cfg['batch_size'] // 4, 4]))

                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                  except:
                      scipy.misc.imsave(
                          os.path.join(SAMPLE_SAVE, "{0}_GAN.png".format(counter)),
                          ren_img[0])
                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))


              if np.mod(counter, 1000) == 1:
                  self.save(config.checkpoint_dir, counter)

  def train_3(self, config):
      global_step = tf.Variable(0, name='global_step', trainable=False)
      self.d_learning_rate = tf.train.exponential_decay(cfg['d_eta'], global_step, cfg['decay_steps'], 0.96, staircase=True)
      self.g_learning_rate = tf.train.exponential_decay(cfg['g_eta'], global_step, cfg['decay_steps'], 0.96, staircase=True)
      if str.lower(str(cfg["adam_D"])) == "true":
          print("Adam")
          d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.d_loss,
                                                                                                          var_list=self.d_vars)
      if str.lower(str(cfg["adam_G"])) == "true":
          print("Adam")
          g_optim = tf.train.AdamOptimizer(self.g_learning_rate, beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.g_loss,
                                                                                                          var_list=self.g_vars)
      self.d_lr_sum = tf.summary.scalar('D_eta', self.d_learning_rate)
      self.g_lr_sum = tf.summary.scalar('G_eta', self.g_learning_rate)

      tf.global_variables_initializer().run()

      shutil.copyfile(sys.argv[1], os.path.join(LOGDIR, 'config.json'))
      self.g_sum = merge_summary([self.d__sum,
                                  self.d_loss_fake_sum, self.g_loss_sum])
      self.d_sum = merge_summary( [self.d_sum, self.d_loss_real_sum, self.d_loss_sum, self.d_lr_sum, self.g_lr_sum])
      self.writer = SummaryWriter(os.path.join(SAMPLE_SAVE, "log"), self.sess.graph)

      # Sample one fixed Z and view parameters to test the sampling
      sample_z = self.sampling_Z()
      if "_3" in str(cfg['view_func']) or "_4" in str(cfg['view_func']):
          sample_azimuth = np.random.uniform(0, 359, cfg['batch_size'])
          sample_view = self.gen_view_func(cfg['batch_size'], sample_azimuth, cfg['ele_low'], cfg['ele_high'],
                                           cfg['scale_low'], cfg['scale_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=to_bool(str(cfg['with_translation'])),
                                           with_scale=to_bool(str(cfg['with_translation'])))
      elif "_5" in str(cfg['view_func']) in str(cfg['view_func']):
          sample_azimuth = np.random.uniform(0, 359, cfg['batch_size'])
          sample_scale = np.random.uniform(cfg['scale_low'], cfg['scale_high'], cfg['batch_size'])
          sample_view = self.gen_view_func(cfg['batch_size'], sample_azimuth, sample_scale,
                                           cfg['ele_low'], cfg['ele_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=to_bool(str(cfg['with_translation'])),
                                           with_scale=to_bool(str(cfg['with_translation'])))
      else:
          sample_view =  self.gen_view_func(cfg['batch_size'], cfg['ele_low'], cfg['ele_high'],
                                                             cfg['azi_low'], cfg['azi_high'],
                                                             cfg['scale_low'], cfg['scale_high'],
                                                             cfg['x_low'], cfg['x_high'],
                                                             cfg['y_low'], cfg['y_high'],
                                                             cfg['z_low'], cfg['z_high'],
                                                             with_translation = to_bool(str(cfg['with_translation'])),
                                                             with_scale = to_bool(str(cfg['with_translation'])))

      sample_files = self.data[0:cfg['batch_size']]

      if config.dataset == "shoes" or config.dataset == "mvc" or config.dataset == "cats" or config.dataset == "cars":
          sample_images = [get_image_mvc(sample_file, self.output_height) for sample_file in sample_files]

      else:
          sample_images = [get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for sample_file in sample_files]
      sample_inputs = np.array(sample_images).astype(np.float32)

      counter = 1
      start_time = time.time()
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")

      self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))

      for epoch in range(cfg['max_epochs']):
          if config.dataset == "cars":
              self.azimuth_all, self.scale_all = create_car_azimuth_scale(cfg['scale_low'], cfg['scale_high'])
          random.shuffle(self.data)
          batch_idxs = min(len(self.data), config.train_size) // cfg['batch_size']

          for idx in range(0, batch_idxs):
              batch_files = self.data[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
              if config.dataset == "shoes" or config.dataset == "mvc" or config.dataset == "cats" or config.dataset == "cars":
                  batch_images = [get_image_mvc(batch_file, self.output_height) for batch_file in batch_files]
              else:
                  batch_images = [get_image(batch_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for batch_file in batch_files]


              #Randomly flip images
              if config.dataset is not "cars" or config.dataset is not "carsOld":
                  if np.random.uniform(0.1, 1.0) <= 0.5:
                      try:
                          batch_images = horizontal_flip(batch_images)
                      except:
                          print("cannot flip")
              batch_z = self.sampling_Z()
              if "_3" in str(cfg['view_func']) or "_4" in str(cfg['view_func']):
                  batch_azimuth = self.azimuth_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_view = self.gen_view_func(cfg['batch_size'], batch_azimuth, cfg['ele_low'], cfg['ele_high'],
                                                   cfg['scale_low'], cfg['scale_high'],
                                                   cfg['x_low'], cfg['x_high'],
                                                   cfg['y_low'], cfg['y_high'],
                                                   cfg['z_low'], cfg['z_high'],
                                                   with_translation=to_bool(str(cfg['with_translation'])),
                                                   with_scale=to_bool(str(cfg['with_translation'])))
              elif "_5" in str(cfg['view_func']) in str(cfg['view_func']):
                  batch_azimuth = self.azimuth_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_scale = self.scale_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_view = self.gen_view_func(cfg['batch_size'], batch_azimuth, batch_scale,
                                                   cfg['ele_low'], cfg['ele_high'],
                                                   cfg['x_low'], cfg['x_high'],
                                                   cfg['y_low'], cfg['y_high'],
                                                   cfg['z_low'], cfg['z_high'],
                                                   with_translation=to_bool(str(cfg['with_translation'])),
                                                   with_scale=to_bool(str(cfg['with_translation'])))
              else:
                  batch_view = self.gen_view_func(cfg['batch_size'], cfg['ele_low'], cfg['ele_high'],
                                                   cfg['azi_low'], cfg['azi_high'],
                                                   cfg['scale_low'], cfg['scale_high'],
                                                   cfg['x_low'], cfg['x_high'],
                                                   cfg['y_low'], cfg['y_high'],
                                                   cfg['z_low'], cfg['z_high'],
                                                   with_translation=to_bool(str(cfg['with_translation'])),
                                                   with_scale=to_bool(str(cfg['with_translation'])))

              feed = {self.inputs: batch_images,
                      self.z: batch_z,
                      self.view_in: batch_view,
                      self.is_training: True}
              try:
                  # Update D network
                  _, summary_str, step_out, lr = self.sess.run([d_optim, self.d_sum, global_step, self.d_learning_rate],
                                                 feed_dict=feed)
                  print("Step {0}: Learning rate {1}".format(step_out, lr))
                  self.writer.add_summary(summary_str, counter)
                  # Update G network
                  _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                 feed_dict=feed)
                  self.writer.add_summary(summary_str, counter)

                  if str.lower(str(cfg["g_twice"])) == "true":
                      print("Train G Twice")
                      # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                      _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                     feed_dict=feed)
                      self.writer.add_summary(summary_str, counter)

                  errD_fake = self.d_loss_fake.eval(feed)
                  errD_real = self.d_loss_real.eval(feed)
                  errG = self.g_loss.eval(feed)
                  errQ = self.q_loss.eval(feed)
              except:
                    print("FAIL")
                    continue



              counter += 1
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                       time.time() - start_time, errD_fake + errD_real, errG, errQ))

              if np.mod(counter, 500) == 1:
                  if self.with_rotation:
                      feed_eval = {self.inputs: sample_inputs,
                                   self.z: sample_z,
                                   self.view_in: sample_view,
                                   self.is_training: False}


                  samples, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict=feed_eval)
                  ren_img = inverse_transform(samples)
                  ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
                  try:
                      scipy.misc.imsave(
                          os.path.join(SAMPLE_SAVE, "{0}_GAN.png".format(counter)),
                          merge(ren_img, [cfg['batch_size'] // 4, 4]))

                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                  except:
                      scipy.misc.imsave(
                          os.path.join(SAMPLE_SAVE, "{0}_GAN.png".format(counter)),
                          ren_img[0])
                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))


              if np.mod(counter, 1000) == 2:
                  self.save(config.checkpoint_dir, counter)

  def train_4(self, config):
      self.d_lr_in = tf.placeholder(tf.float32, None, name='d_eta')
      self.g_lr_in = tf.placeholder(tf.float32, None, name='d_eta')

      if str.lower(str(cfg["adam_D"])) == "true":
          print("Adam")
          d_optim = tf.train.AdamOptimizer(self.d_lr_in, beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.d_loss, var_list=self.d_vars)
      else:
          print("SGD")
          d_optim = tf.train.MomentumOptimizer(self.d_lr_in, -0.3).minimize(self.d_loss, var_list=self.d_vars)

      if str.lower(str(cfg["adam_G"])) == "true":
          print("Adam")
          g_optim = tf.train.AdamOptimizer(self.g_lr_in, beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.g_loss, var_list=self.g_vars)
      else:
          print("SGD")
          g_optim = tf.train.MomentumOptimizer(self.g_lr_in, -0.3).minimize(self.g_loss, var_list=self.g_vars)

      tf.global_variables_initializer().run()

      shutil.copyfile(sys.argv[1], os.path.join(LOGDIR, 'config.json'))
      self.g_sum = merge_summary([self.d__sum,
                                  self.d_loss_fake_sum, self.g_loss_sum])
      self.d_sum = merge_summary( [self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
      self.writer = SummaryWriter(os.path.join(SAMPLE_SAVE, "log"), self.sess.graph)

      # Sample one fixed Z and view parameters to test the sampling
      sample_z = self.sampling_Z()
      if "_3" in str(cfg['view_func']) or "_4" in str(cfg['view_func']):
          sample_azimuth = np.random.uniform(0, 359, cfg['batch_size'])
          sample_view = self.gen_view_func(cfg['batch_size'], sample_azimuth, cfg['ele_low'], cfg['ele_high'],
                                           cfg['scale_low'], cfg['scale_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=to_bool(str(cfg['with_translation'])),
                                           with_scale=to_bool(str(cfg['with_translation'])))
      elif "_5" in str(cfg['view_func']) in str(cfg['view_func']):
          sample_azimuth = np.random.uniform(0, 359, cfg['batch_size'])
          sample_scale = np.random.uniform(cfg['scale_low'], cfg['scale_high'], cfg['batch_size'])
          sample_view = self.gen_view_func(cfg['batch_size'], sample_azimuth, sample_scale,
                                           cfg['ele_low'], cfg['ele_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=to_bool(str(cfg['with_translation'])),
                                           with_scale=to_bool(str(cfg['with_translation'])))
      else:
          sample_view =  self.gen_view_func(cfg['batch_size'], cfg['ele_low'], cfg['ele_high'],
                                                             cfg['azi_low'], cfg['azi_high'],
                                                             cfg['scale_low'], cfg['scale_high'],
                                                             cfg['x_low'], cfg['x_high'],
                                                             cfg['y_low'], cfg['y_high'],
                                                             cfg['z_low'], cfg['z_high'],
                                                             with_translation = to_bool(str(cfg['with_translation'])),
                                                             with_scale = to_bool(str(cfg['with_translation'])))

      sample_files = self.data[0:cfg['batch_size']]

      if config.dataset == "shoes" or config.dataset == "mvc" or config.dataset == "cats" or config.dataset == "cars":
          sample_images = [get_image_mvc(sample_file, self.output_height) for sample_file in sample_files]

      else:
          sample_images = [get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for sample_file in sample_files]
      sample_inputs = np.array(sample_images).astype(np.float32)

      counter = 1
      start_time = time.time()
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")

      self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))
      d_lr = cfg['d_eta']
      g_lr = cfg['g_eta']
      for epoch in range(cfg['max_epochs']):
          d_lr = d_lr if epoch < cfg['epoch_step'] else d_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'] + 1)
          g_lr = g_lr if epoch < cfg['epoch_step'] else g_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'] + 1)

          if config.dataset == "cars":
              self.azimuth_all, self.scale_all = create_car_azimuth_scale(cfg['scale_low'], cfg['scale_high'])
          random.shuffle(self.data)
          batch_idxs = min(len(self.data), config.train_size) // cfg['batch_size']

          for idx in range(0, batch_idxs):
              batch_files = self.data[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
              if config.dataset == "shoes" or config.dataset == "mvc" or config.dataset == "cats" or config.dataset == "cars":
                  batch_images = [get_image_mvc(batch_file, self.output_height) for batch_file in batch_files]
              else:
                  batch_images = [get_image(batch_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for batch_file in batch_files]


              #Randomly flip images
              if config.dataset is not "cars" or config.dataset is not "carsOld":
                  if np.random.uniform(0.1, 1.0) <= 0.5:
                      try:
                          batch_images = horizontal_flip(batch_images)
                      except:
                          print("cannot flip")
              batch_z = self.sampling_Z()
              if "_3" in str(cfg['view_func']) or "_4" in str(cfg['view_func']):
                  batch_azimuth = self.azimuth_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_view = self.gen_view_func(cfg['batch_size'], batch_azimuth, cfg['ele_low'], cfg['ele_high'],
                                                  cfg['scale_low'], cfg['scale_high'],
                                                  cfg['x_low'], cfg['x_high'],
                                                  cfg['y_low'], cfg['y_high'],
                                                  cfg['z_low'], cfg['z_high'],
                                                  with_translation=to_bool(str(cfg['with_translation'])),
                                                  with_scale=to_bool(str(cfg['with_translation'])))
              elif "_5" in str(cfg['view_func']) in str(cfg['view_func']):
                  batch_azimuth = self.azimuth_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_scale = self.scale_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_view = self.gen_view_func(cfg['batch_size'], batch_azimuth, batch_scale,
                                                   cfg['ele_low'], cfg['ele_high'],
                                                   cfg['x_low'], cfg['x_high'],
                                                   cfg['y_low'], cfg['y_high'],
                                                   cfg['z_low'], cfg['z_high'],
                                                   with_translation=to_bool(str(cfg['with_translation'])),
                                                   with_scale=to_bool(str(cfg['with_translation'])))
              else:
                  batch_view = self.gen_view_func(cfg['batch_size'], cfg['ele_low'], cfg['ele_high'],
                                                  cfg['azi_low'], cfg['azi_high'],
                                                  cfg['scale_low'], cfg['scale_high'],
                                                  cfg['x_low'], cfg['x_high'],
                                                  cfg['y_low'], cfg['y_high'],
                                                  cfg['z_low'], cfg['z_high'],
                                                  with_translation=to_bool(str(cfg['with_translation'])),
                                                  with_scale=to_bool(str(cfg['with_translation'])))

              feed = {self.inputs: batch_images,
                      self.z: batch_z,
                      self.view_in: batch_view,
                      self.d_lr_in: d_lr,
                      self.g_lr_in: g_lr,
                      self.is_training: True}
              # try:
              # Update D network
              _, summary_str = self.sess.run([d_optim, self.d_sum],
                                             feed_dict=feed)
              self.writer.add_summary(summary_str, counter)

              if str.lower(str(cfg["d_twice"])) == "true":
                  print("Train D Twice")
                  _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                 feed_dict=feed)
                  self.writer.add_summary(summary_str, counter)
              # Update G network
              _, summary_str = self.sess.run([g_optim, self.g_sum],
                                             feed_dict=feed)
              self.writer.add_summary(summary_str, counter)
              print("Learning rate " + str(d_lr))
              if str.lower(str(cfg["g_twice"])) == "true":
                  print("Train G Twice")
                  # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                  _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                 feed_dict=feed)
                  self.writer.add_summary(summary_str, counter)

              errD_fake = self.d_loss_fake.eval(feed)
              errD_real = self.d_loss_real.eval(feed)
              errG = self.g_loss.eval(feed)
              errQ = self.q_loss.eval(feed)
              # except:
              #       print("FAIL")
              #       continue



              counter += 1
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                       time.time() - start_time, errD_fake + errD_real, errG, errQ))

              if np.mod(counter, 500) == 1:
                  if self.with_rotation:
                      feed_eval = {self.inputs: sample_inputs,
                                   self.z: sample_z,
                                   self.view_in: sample_view,
                                   self.d_lr_in: d_lr,
                                   self.g_lr_in: g_lr,
                                   self.is_training: False}


                  samples, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict=feed_eval)
                  ren_img = inverse_transform(samples)
                  ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
                  try:
                      scipy.misc.imsave(
                          os.path.join(SAMPLE_SAVE, "{0}_GAN.png".format(counter)),
                          merge(ren_img, [cfg['batch_size'] // 4, 4]))

                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                  except:
                      scipy.misc.imsave(
                          os.path.join(SAMPLE_SAVE, "{0}_GAN.png".format(counter)),
                          ren_img[0])
                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))


              if np.mod(counter, 1000) == 1:
                  self.save(config.checkpoint_dir, counter)

  def train_5(self, config):
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      self.d_learning_rate = tf.train.exponential_decay(cfg['d_eta'], self.global_step, cfg['decay_steps'], cfg['decay_rate'], staircase=True)
      self.g_learning_rate = tf.train.exponential_decay(cfg['g_eta'], self.global_step, cfg['decay_steps'], cfg['decay_rate'], staircase=True)
      if str.lower(str(cfg["adam_D"])) == "true":
          print("Adam")
          d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.d_loss,
                                                                                                          var_list=self.d_vars, global_step=self.global_step)
      if str.lower(str(cfg["adam_G"])) == "true":
          print("Adam")
          g_optim = tf.train.AdamOptimizer(self.g_learning_rate, beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.g_loss,
                                                                                                          var_list=self.g_vars, global_step=self.global_step)
      self.d_lr_sum = tf.summary.scalar('D_eta', self.d_learning_rate)
      self.g_lr_sum = tf.summary.scalar('G_eta', self.g_learning_rate)

      tf.global_variables_initializer().run()

      shutil.copyfile(sys.argv[1], os.path.join(LOGDIR, 'config.json'))
      self.g_sum = merge_summary([self.d__sum,
                                  self.d_loss_fake_sum, self.g_loss_sum])
      self.d_sum = merge_summary( [self.d_sum, self.d_loss_real_sum, self.d_loss_sum, self.d_lr_sum, self.g_lr_sum])
      self.writer = SummaryWriter(os.path.join(SAMPLE_SAVE, "log"), self.sess.graph)

      # Sample one fixed Z and view parameters to test the sampling
      sample_z = self.sampling_Z()
      if "_3" in str(cfg['view_func']) or "_4" in str(cfg['view_func']):
          sample_azimuth = np.random.uniform(0, 359, cfg['batch_size'])
          sample_view = self.gen_view_func(cfg['batch_size'], sample_azimuth, cfg['ele_low'], cfg['ele_high'],
                                           cfg['scale_low'], cfg['scale_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=to_bool(str(cfg['with_translation'])),
                                           with_scale=to_bool(str(cfg['with_translation'])))
      elif "_5" in str(cfg['view_func']) in str(cfg['view_func']):
          sample_azimuth = np.random.uniform(0, 359, cfg['batch_size'])
          sample_scale = np.random.uniform(cfg['scale_low'], cfg['scale_high'], cfg['batch_size'])
          sample_view = self.gen_view_func(cfg['batch_size'], sample_azimuth, sample_scale,
                                           cfg['ele_low'], cfg['ele_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=to_bool(str(cfg['with_translation'])),
                                           with_scale=to_bool(str(cfg['with_translation'])))
      else:
          sample_view =  self.gen_view_func(cfg['batch_size'], cfg['ele_low'], cfg['ele_high'],
                                                             cfg['azi_low'], cfg['azi_high'],
                                                             cfg['scale_low'], cfg['scale_high'],
                                                             cfg['x_low'], cfg['x_high'],
                                                             cfg['y_low'], cfg['y_high'],
                                                             cfg['z_low'], cfg['z_high'],
                                                             with_translation = to_bool(str(cfg['with_translation'])),
                                                             with_scale = to_bool(str(cfg['with_translation'])))

      sample_files = self.data[0:cfg['batch_size']]

      if config.dataset == "shoes" or config.dataset == "mvc" or config.dataset == "cats" or config.dataset == "cars":
          sample_images = [get_image_mvc(sample_file, self.output_height) for sample_file in sample_files]

      else:
          sample_images = [get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for sample_file in sample_files]
      sample_inputs = np.array(sample_images).astype(np.float32)

      counter = 1
      start_time = time.time()
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")

      self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))

      for epoch in range(cfg['max_epochs']):
          if config.dataset == "cars":
              self.azimuth_all, self.scale_all = create_car_azimuth_scale(cfg['scale_low'], cfg['scale_high'])
          random.shuffle(self.data)
          batch_idxs = min(len(self.data), config.train_size) // cfg['batch_size']

          for idx in range(0, batch_idxs):
              batch_files = self.data[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
              if config.dataset == "shoes" or config.dataset == "mvc" or config.dataset == "cats" or config.dataset == "cars":
                  batch_images = [get_image_mvc(batch_file, self.output_height) for batch_file in batch_files]
              else:
                  batch_images = [get_image(batch_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for batch_file in batch_files]


              #Randomly flip images
              if config.dataset is not "cars" or config.dataset is not "carsOld":
                  if np.random.uniform(0.1, 1.0) <= 0.5:
                      try:
                          batch_images = horizontal_flip(batch_images)
                      except:
                          print("cannot flip")
              batch_z = self.sampling_Z()
              if "_3" in str(cfg['view_func']) or "_4" in str(cfg['view_func']):
                  batch_azimuth = self.azimuth_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_view = self.gen_view_func(cfg['batch_size'], batch_azimuth, cfg['ele_low'], cfg['ele_high'],
                                                   cfg['scale_low'], cfg['scale_high'],
                                                   cfg['x_low'], cfg['x_high'],
                                                   cfg['y_low'], cfg['y_high'],
                                                   cfg['z_low'], cfg['z_high'],
                                                   with_translation=to_bool(str(cfg['with_translation'])),
                                                   with_scale=to_bool(str(cfg['with_translation'])))
              elif "_5" in str(cfg['view_func']) in str(cfg['view_func']):
                  batch_azimuth = self.azimuth_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_scale = self.scale_all[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
                  batch_view = self.gen_view_func(cfg['batch_size'], batch_azimuth, batch_scale,
                                                   cfg['ele_low'], cfg['ele_high'],
                                                   cfg['x_low'], cfg['x_high'],
                                                   cfg['y_low'], cfg['y_high'],
                                                   cfg['z_low'], cfg['z_high'],
                                                   with_translation=to_bool(str(cfg['with_translation'])),
                                                   with_scale=to_bool(str(cfg['with_translation'])))
              else:
                  batch_view = self.gen_view_func(cfg['batch_size'], cfg['ele_low'], cfg['ele_high'],
                                                   cfg['azi_low'], cfg['azi_high'],
                                                   cfg['scale_low'], cfg['scale_high'],
                                                   cfg['x_low'], cfg['x_high'],
                                                   cfg['y_low'], cfg['y_high'],
                                                   cfg['z_low'], cfg['z_high'],
                                                   with_translation=to_bool(str(cfg['with_translation'])),
                                                   with_scale=to_bool(str(cfg['with_translation'])))

              feed = {self.inputs: batch_images,
                      self.z: batch_z,
                      self.view_in: batch_view,
                      self.is_training: True}

              # Update D network
              _, summary_str, step_out, lr = self.sess.run([d_optim, self.d_sum, self.global_step, self.d_learning_rate],
                                             feed_dict=feed)
              print("Step {0}: Learning rate {1}".format(step_out, lr))
              self.writer.add_summary(summary_str, counter)
              # Update G network
              _, summary_str = self.sess.run([g_optim, self.g_sum],
                                             feed_dict=feed)
              self.writer.add_summary(summary_str, counter)

              if str.lower(str(cfg["g_twice"])) == "true":
                  print("Train G Twice")
                  # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                  _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                 feed_dict=feed)
                  self.writer.add_summary(summary_str, counter)

              errD_fake = self.d_loss_fake.eval(feed)
              errD_real = self.d_loss_real.eval(feed)
              errG = self.g_loss.eval(feed)
              errQ = self.q_loss.eval(feed)

              counter += 1
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                    % (epoch, step_out, batch_idxs,
                       time.time() - start_time, errD_fake + errD_real, errG, errQ))

              if np.mod(counter, 500) == 1:
                  if self.with_rotation:
                      feed_eval = {self.inputs: sample_inputs,
                                   self.z: sample_z,
                                   self.view_in: sample_view,
                                   self.is_training: False}


                  samples, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict=feed_eval)
                  ren_img = inverse_transform(samples)
                  ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
                  try:
                      scipy.misc.imsave(
                          os.path.join(SAMPLE_SAVE, "{0}_GAN.png".format(counter)),
                          merge(ren_img, [cfg['batch_size'] // 4, 4]))

                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                  except:
                      scipy.misc.imsave(
                          os.path.join(SAMPLE_SAVE, "{0}_GAN.png".format(counter)),
                          ren_img[0])
                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))


              if np.mod(counter, 1000) == 2:
                  self.save(config.checkpoint_dir, counter)

  def discriminator_IN_with_latent_variable(self, image, cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)
      #Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe

      batch_size = tf.shape(image)[0]
      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          if not self.y_dim:
              h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
              h1 = lrelu(self.instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
              h2 = lrelu(self.instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
              h3 = lrelu(self.instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))

              #Returning logits to determine whether the images are real or fake
              h4 = linear(slim.flatten(h3), 1, 'd_h4_lin')

              # Recognition network for latent variables has an additional layer
              encoder = lrelu(self.d_bn4(linear(slim.flatten(h3), 128, 'd_latent')))
              # Compute mean and variance for Gaussian posterior of continuous latents
              cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

              return tf.nn.sigmoid(h4), h4, tf.nn.tanh(cont_vars)

  def discriminator_IN_clean(self, image,  cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          if not self.y_dim:
              h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
              h1 = lrelu(self.instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
              h2 = lrelu(self.instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
              h3 = lrelu(self.instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))

              #Returning logits to determine whether the images are real or fake
              h4 = linear(slim.flatten(h3), 1, 'd_h4_lin')

              # Recognition network for latent variables has an additional layer
              encoder = lrelu(self.d_bn4(linear(slim.flatten(h3), 128, 'd_latent')))
              # Compute mean and variance for Gaussian posterior of continuous latents
              cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

              return tf.nn.sigmoid(h4), h4, tf.nn.tanh(cont_vars)

  def discriminator_IN_clean_noBN(self, image,  cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          if not self.y_dim:
              h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
              h1 = lrelu(self.instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
              h2 = lrelu(self.instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
              h3 = lrelu(self.instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))

              #Returning logits to determine whether the images are real or fake
              h4 = linear(slim.flatten(h3), 1, 'd_h4_lin')

              # Recognition network for latent variables has an additional layer
              encoder = lrelu((linear(slim.flatten(h3), 128, 'd_latent')))
              # Compute mean and variance for Gaussian posterior of continuous latents
              cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

              return tf.nn.sigmoid(h4), h4, tf.nn.tanh(cont_vars)

  def discriminator_IN_clean_noBN_res128(self, image,  cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          if not self.y_dim:
              h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
              h1 = lrelu(self.instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
              h2 = lrelu(self.instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
              h3 = lrelu(self.instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))
              h4 = lrelu(self.instance_norm(conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv'), 'd_in4'))
              #Returning logits to determine whether the images are real or fake
              h5 = linear(slim.flatten(h4), 1, 'd_h5_lin')

              # Recognition network for latent variables has an additional layer
              encoder = lrelu((linear(slim.flatten(h4), 128, 'd_latent')))
              # Compute mean and variance for Gaussian posterior of continuous latents
              cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

              return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars)

  def discriminator_IN_clean_noBN_res128_specNorm(self, image,  cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          if not self.y_dim:
              h0 = lrelu(conv2d_specNorm(image, self.df_dim, name='d_h0_conv'))
              h1 = lrelu(self.instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
              h2 = lrelu(self.instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))

              h3 = lrelu(self.instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))
              h4 = lrelu(self.instance_norm(conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv'), 'd_in4'))
              #Returning logits to determine whether the images are real or fake
              h5 = linear_specNorm(slim.flatten(h4), 1, 'd_h5_lin')

              # Recognition network for latent variables has an additional layer
              encoder = lrelu((linear_specNorm(slim.flatten(h4), 128, 'd_latent')))
              # Compute mean and variance for Gaussian posterior of continuous latents
              cont_vars = linear_specNorm(encoder, cont_dim, "d_latent_prediction")

              return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars)

  def discriminator_IN_clean_noBN_res128_specNorm_2(self, image,  cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          if not self.y_dim:
              h0 = lrelu(conv2d_specNorm(image, self.df_dim, name='d_h0_conv'))
              h1 = lrelu(self.instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
              h2 = lrelu(self.instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
              h3 = lrelu(self.instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))
              h4 = lrelu(self.instance_norm(conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv'), 'd_in4'))
              #Returning logits to determine whether the images are real or fake
              h5 = linear_specNorm(slim.flatten(h4), 1, 'd_h5_lin')

              # Recognition network for latent variables has an additional layer
              encoder = lrelu((linear_specNorm(slim.flatten(h4), 256, 'd_latent')))
              # Compute mean and variance for Gaussian posterior of continuous latents
              cont_vars = linear_specNorm(encoder, cont_dim, "d_latent_prediction")

              return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars)

  def discriminator_IN_style_noBN_res128(self, image,  cont_dim, reuse=False):
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = self.instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          d_h1_style = tf.concat([h1_mean, h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = self.instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          d_h2_style = tf.concat([h2_mean, h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = self.instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          d_h3_style = tf.concat([h3_mean, h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          h4 = conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv')
          h4, h4_mean, h4_var = self.instance_norm(h4, 'd_in4', True)
          h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          d_h4_style = tf.concat([h4_mean, h4_var], 0)
          d_h4, d_h4_logits = self.linear_classifier(d_h4_style, "d_h4_class")
          h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear(slim.flatten(h4), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear(slim.flatten(h4), 128, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits

  def discriminator_IN_style_noBN_res128_log(self, image,  cont_dim, reuse=False):
      epsilon = 1e-10
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = self.instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          log_h1_var = tf.log(h1_var + epsilon)
          d_h1_style = tf.concat([h1_mean, log_h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = self.instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          log_h2_var = tf.log(h2_var + epsilon)
          d_h2_style = tf.concat([h2_mean, log_h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = self.instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          log_h3_var = tf.log(h3_var + epsilon)
          d_h3_style = tf.concat([h3_mean, log_h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          h4 = conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv')
          h4, h4_mean, h4_var = self.instance_norm(h4, 'd_in4', True)
          h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          log_h4_var = tf.log(h4_var + epsilon)
          d_h4_style = tf.concat([h4_mean, log_h4_var], 0)
          d_h4, d_h4_logits = self.linear_classifier(d_h4_style, "d_h4_class")
          h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear(slim.flatten(h4), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear(slim.flatten(h4), 128, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits

  def discriminator_IN_style_noBN_res64(self, image,  cont_dim, reuse=False):
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = self.instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          d_h1_style = tf.concat([h1_mean, h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = self.instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          d_h2_style = tf.concat([h2_mean, h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = self.instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          d_h3_style = tf.concat([h3_mean, h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

           #Returning logits to determine whether the images are real or fake
          h5 = linear(slim.flatten(h3), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear(slim.flatten(h3 ), 128, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits

  def discriminator_IN_style_noBN_res128_specNorm(self, image,  cont_dim, reuse=False):
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d_specNorm(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = self.instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          d_h1_style = tf.concat([h1_mean, h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = self.instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          d_h2_style = tf.concat([h2_mean, h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = self.instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          d_h3_style = tf.concat([h3_mean, h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          h4 = conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv')
          h4, h4_mean, h4_var = self.instance_norm(h4, 'd_in4', True)
          h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          d_h4_style = tf.concat([h4_mean, h4_var], 0)
          d_h4, d_h4_logits = self.linear_classifier(d_h4_style, "d_h4_class")
          h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear_specNorm(slim.flatten(h4), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear_specNorm(slim.flatten(h4), 128, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear_specNorm(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits

  def discriminator_IN_style_noBN_res128_specNorm_log(self, image,  cont_dim, reuse=False):
      epsilon = 1e-10
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d_specNorm(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = self.instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          log_h1_var = tf.log(h1_var + epsilon)
          d_h1_style = tf.concat([h1_mean, log_h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = self.instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          log_h2_var = tf.log(h2_var + epsilon)
          d_h2_style = tf.concat([h2_mean, log_h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = self.instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          log_h3_var = tf.log(h3_var + epsilon)
          d_h3_style = tf.concat([h3_mean, log_h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          h4 = conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv')
          h4, h4_mean, h4_var = self.instance_norm(h4, 'd_in4', True)
          h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          log_h4_var = tf.log(h4_var + epsilon)
          d_h4_style = tf.concat([h4_mean, log_h4_var], 0)
          d_h4, d_h4_logits = self.linear_classifier(d_h4_style, "d_h4_class")
          h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear_specNorm(slim.flatten(h4), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear_specNorm(slim.flatten(h4), 128, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear_specNorm(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits

  def discriminator_IN_style_noBN_res128_specNorm_2_log(self, image,  cont_dim, reuse=False):
      epsilon = 1e-10
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d_specNorm(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = self.instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          log_h1_var = tf.log(h1_var + epsilon)
          d_h1_style = tf.concat([h1_mean, log_h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = self.instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          log_h2_var = tf.log(h2_var + epsilon)
          d_h2_style = tf.concat([h2_mean, log_h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = self.instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          log_h3_var = tf.log(h3_var + epsilon)
          d_h3_style = tf.concat([h3_mean, log_h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          h4 = conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv')
          h4, h4_mean, h4_var = self.instance_norm(h4, 'd_in4', True)
          h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          log_h4_var = tf.log(h4_var + epsilon)
          d_h4_style = tf.concat([h4_mean, log_h4_var], 0)
          d_h4, d_h4_logits = self.linear_classifier(d_h4_style, "d_h4_class")
          h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear_specNorm(slim.flatten(h4), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear_specNorm(slim.flatten(h4), 256, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear_specNorm(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits

  def discriminator_IN_style_noBN_res128_specNorm_2(self, image,  cont_dim, reuse=False):
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d_specNorm(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = self.instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          d_h1_style = tf.concat([h1_mean, h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = self.instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          d_h2_style = tf.concat([h2_mean, h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = self.instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          d_h3_style = tf.concat([h3_mean, h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          h4 = conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv')
          h4, h4_mean, h4_var = self.instance_norm(h4, 'd_in4', True)
          h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          d_h4_style = tf.concat([h4_mean, h4_var], 0)
          d_h4, d_h4_logits = self.linear_classifier(d_h4_style, "d_h4_class")
          h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear_specNorm(slim.flatten(h4), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear_specNorm(slim.flatten(h4), 256, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear_specNorm(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits

  def discriminator_IN_style_noBN_res128_2(self, image,  cont_dim, reuse=False):
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = self.instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          d_h1_style = tf.concat([h1_mean, h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = self.instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          d_h2_style = tf.concat([h2_mean, h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = self.instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          d_h3_style = tf.concat([h3_mean, h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          h4 = conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv')
          h4, h4_mean, h4_var = self.instance_norm(h4, 'd_in4', True)
          h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          d_h4_style = tf.concat([h4_mean, h4_var], 0)
          d_h4, d_h4_logits = self.linear_classifier(d_h4_style, "d_h4_class")
          h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear(slim.flatten(h4), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear(slim.flatten(h4), 512, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits

  def discriminator_IN_style_noBN_res128_wscale(self, image,  cont_dim, reuse=False):
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d_specNorm_wscale(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm_wscale(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = self.instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          d_h1_style = tf.concat([h1_mean, h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier_wscale(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm_wscale(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = self.instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          d_h2_style = tf.concat([h2_mean, h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier_wscale(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm_wscale(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = self.instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          d_h3_style = tf.concat([h3_mean, h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier_wscale(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          h4 = conv2d_specNorm_wscale(h3, self.df_dim * 16, name='d_h4_conv')
          h4, h4_mean, h4_var = self.instance_norm(h4, 'd_in4', True)
          h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          d_h4_style = tf.concat([h4_mean, h4_var], 0)
          d_h4, d_h4_logits = self.linear_classifier_wscale(d_h4_style, "d_h4_class")
          h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear_wscale(slim.flatten(h4), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear_wscale(slim.flatten(h4), 128, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear_wscale(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits

  def discriminator_IN_clean_noBN_noSpecNorm(self, image,  cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()


          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
          h1 = lrelu(self.instance_norm(conv2d(h0, self.df_dim * 2, name='d_h1_conv'), 'd_in1'))
          h2 = lrelu(self.instance_norm(conv2d(h1, self.df_dim * 4, name='d_h2_conv'), 'd_in2'))
          h3 = lrelu(self.instance_norm(conv2d(h2, self.df_dim * 8, name='d_h3_conv'), 'd_in3'))

          #Returning logits to determine whether the images are real or fake
          h4 = linear(slim.flatten(h3), 1, 'd_h4_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear(slim.flatten(h3), 128, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h4), h4, tf.nn.tanh(cont_vars)

  def discriminator_specNorm_clean_noBN(self, image, cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          if not self.y_dim:
              h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
              h1 = lrelu(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'))
              h2 = lrelu(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'))
              h3 = lrelu(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'))

              # Returning logits to determine whether the images are real or fake
              h4 = linear(slim.flatten(h3), 1, 'd_h4_lin')

              # Recognition network for latent variables has an additional layer
              encoder = lrelu((linear(slim.flatten(h3), 128, 'd_latent')))
              # Compute mean and variance for Gaussian posterior of continuous latents
              cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

              return tf.nn.sigmoid(h4), h4, tf.nn.tanh(cont_vars)

  def discriminator_IN_clean_noBN_pose_regressor(self, image,  cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          if not self.y_dim:
              h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
              h1 = lrelu(self.instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
              h2 = lrelu(self.instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
              h3 = lrelu(self.instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))

              #Returning logits to determine whether the images are real or fake
              h4 = linear(slim.flatten(h3), 1, 'd_h4_lin')

              # Recognition network for latent variables has an additional layer
              encoder = tf.nn.relu((linear(slim.flatten(h3), 128, 'd_latent')))
              # Compute mean and variance for Gaussian posterior of continuous latents
              cont_vars = linear(encoder, 6, "d_latent_prediction")

              return tf.nn.sigmoid(h4), h4, cont_vars

  def discriminator_IN_clean_res128(self, image,  cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          if not self.y_dim:
              h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
              h1 = lrelu(self.instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
              h2 = lrelu(self.instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
              h3 = lrelu(self.instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))
              h4 = lrelu(self.instance_norm(conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv'), 'd_in4'))

              #Returning logits to determine whether the images are real or fake
              h5 = linear(slim.flatten(h4), 1, 'd_h5_lin')

              # Recognition network for latent variables has an additional layer
              encoder = lrelu(linear(slim.flatten(h3), 128, 'd_latent'))
              # Compute mean and variance for Gaussian posterior of continuous latents
              cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

              return tf.nn.sigmoid(h5),  h5, tf.nn.tanh(cont_vars)

  def discriminator_IN_clean_res128_fixedINFOGAN(self, image,  cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          if not self.y_dim:
              h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
              h1 = lrelu(self.instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
              h2 = lrelu(self.instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
              h3 = lrelu(self.instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))
              h4 = lrelu(self.instance_norm(conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv'), 'd_in4'))

              #Returning logits to determine whether the images are real or fake
              h5 = linear(slim.flatten(h4), 1, 'd_h5_lin')

              # Recognition network for latent variables has an additional layer
              encoder = lrelu(linear(slim.flatten(h4), 128, 'd_latent'))
              # Compute mean and variance for Gaussian posterior of continuous latents
              cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

              return tf.nn.sigmoid(h5),  h5, tf.nn.tan

  def generator_AdaIN(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_d=3, k_w=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_d=3, k_w=3, name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_nozConv2d(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_d=3, k_w=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_d=3, k_w=3, name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          # h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_attention(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_d=3, k_w=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_d=3, k_w=3, name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          h2 = self.attention_3d(h2, self.gf_dim)
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_no_rotation(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_d=3, k_w=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_d=3, k_w=3, name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)


          # Collapsing Z dimension
          h2_2d = tf.reshape(h2, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_lRelu(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_d=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_d=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_d=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_deeper(self, z, view_in, reuse=False):
      # Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          # A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8],
                                  initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0),
                               (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim * 2])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 * 2], k_h=1, k_w=1, d_h=1,
                                              d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4 = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 8], name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4 = self.AdaIn(h4, s4, b4)
          h4 = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim*2], name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_moreConv3d(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h1_2 = deconv3d(h1, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], name='g_h1_2', d_h=1, d_w=1, d_d=1)  # n_filter = 256
          s1_2, b1_2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1_2')
          h1_2 = self.AdaIn(h1_2, s1_2, b1_2)
          h1_2 = tf.nn.relu(h1_2)

          h2 = deconv3d(h1_2, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          h2_2 = deconv3d(h2, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], name='g_h2_2', d_h=1, d_w=1, d_d=1)  # n_filter = 256
          s2_2, b2_2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_2')
          h2_2 = self.AdaIn(h2_2, s2_2, b2_2)
          h2_2 = tf.nn.relu(h2_2)

          h2_rotated = tf_rotation_resampling(h2_2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_moreConv3d_convWithoutINBeforeProjection(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h1_2 = deconv3d(h1, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1_2', d_h=1, d_w=1, d_d=1)  # n_filter = 256
          s1_2, b1_2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1_2')
          h1_2 = self.AdaIn(h1_2, s1_2, b1_2)
          h1_2 = tf.nn.relu(h1_2)

          h2 = deconv3d(h1_2, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          h2_2 = deconv3d(h2, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_2', d_h=1, d_w=1, d_d=1)  # n_filter = 256
          s2_2, b2_2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_2')
          h2_2 = self.AdaIn(h2_2, s2_2, b2_2)
          h2_2 = tf.nn.relu(h2_2)

          # =============================================================================================================
          h2_rotated = tf_rotation_resampling(h2_2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1,
                              d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          # s2_proj1, b2_proj1 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_proj1')
          # h2_proj1 = self.AdaIn(h2_proj1, s2_proj1, b2_proj1)
          h2_proj1 = tf.nn.relu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1,
                              d_d=1, name='g_h2_proj2')  # n_filter = 64
          # s2_proj2, b2_proj2 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z2_proj2')
          # h2_proj2 = self.AdaIn(h2_proj2, s2_proj2, b2_proj2)
          h2_proj2 = tf.nn.relu(h2_proj2)

          # =============================================================================================================

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_moreConv3d_moreConv2D_convWithoutINBeforeProjection(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h1_2 = deconv3d(h1, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1_2', d_h=1, d_w=1, d_d=1)  # n_filter = 256
          s1_2, b1_2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1_2')
          h1_2 = self.AdaIn(h1_2, s1_2, b1_2)
          h1_2 = tf.nn.relu(h1_2)

          h2 = deconv3d(h1_2, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          h2_2 = deconv3d(h2, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_2', d_h=1, d_w=1, d_d=1)  # n_filter = 256
          s2_2, b2_2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_2')
          h2_2 = self.AdaIn(h2_2, s2_2, b2_2)
          h2_2 = tf.nn.relu(h2_2)

          # =============================================================================================================
          h2_rotated = tf_rotation_resampling(h2_2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1,
                              d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          # s2_proj1, b2_proj1 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_proj1')
          # h2_proj1 = self.AdaIn(h2_proj1, s2_proj1, b2_proj1)
          h2_proj1 = tf.nn.relu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1,
                              d_d=1, name='g_h2_proj2')  # n_filter = 64
          # s2_proj2, b2_proj2 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z2_proj2')
          # h2_proj2 = self.AdaIn(h2_proj2, s2_proj2, b2_proj2)
          h2_proj2 = tf.nn.relu(h2_proj2)

          # =============================================================================================================

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=3, k_w=3, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h4_2  = deconv2d(h4, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=3, k_w=3, d_h=1, d_w=1, name='g_h4_2')
          s4_2, b4_2 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4_2')
          h4_2  = self.AdaIn(h4_2, s4_2, b4_2)
          h4_2  = tf.nn.relu(h4_2)

          h5 = deconv2d(h4_2, [batch_size, s_h, s_w, self.gf_dim], k_h=3, k_w=3, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h5_2 = deconv2d(h5, [batch_size, s_h, s_w, self.gf_dim], k_h=3, k_w=3, d_h=1, d_w=1, name='g_h5_2')
          s5_2, b5_2 = self.z_mapping_function(z, self.gf_dim, 'g_z5_2')
          h5_2 = self.AdaIn(h5_2, s5_2, b5_2)
          h5_2 = tf.nn.relu(h5_2)

          h6 = deconv2d(h5_2, [batch_size, s_h, s_w, self.c_dim], d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_res128(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_withBackground(self, z1, z2, z3, view_in, view_in_redundant, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8 // 2, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2 // 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1 // 2, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2 // 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1 // 2, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              # h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          h2_all = tf.concat([h2_rotated, h2], -1)
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_all, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_withBackground_concat(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8 // 2, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2 // 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1 //2, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2 // 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1 // 2, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in2, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          h2_all = tf.concat([h2_rotated, h2], -1)
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_all, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_withBackground_concat_convAfterProjection(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8 // 2, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2 // 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1 //2, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2 // 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1 // 2, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in2, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          h2_all = tf.concat([h2_rotated, h2], -1)

          h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu( h2_proj1)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj1, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_withBackgroundSmaller_concat_convAfterProjection(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2 // 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1 // 2, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in2, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          h2_all = tf.concat([h2_rotated, h2], -1)

          h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu( h2_proj1)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj1, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_withBackgroundSmaller_concat_convAfterProjection_noZ3(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2 // 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1 // 2, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in2, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          h2_all = tf.concat([h2_rotated, h2], -1)

          h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu( h2_proj1)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj1, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          # h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          # s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          # h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_withBackgroundSmaller_concat_convAfterProjection1x1_noZ3(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z3, self.gf_dim * 2 // 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z3, self.gf_dim * 1 // 2, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in3, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          h2_all = tf.concat([h2_rotated, h2], -1)

          h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=1, k_w=1, k_d=1, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu( h2_proj1)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj1, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          # h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          # s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          # h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_concatDepth_convAfterProjection_noZ3(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in2, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          h2_all = tf.concat([h2_rotated, h2], -2)

          h2_proj1 = conv3d(h2_all, self.gf_dim * 1, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=2, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu( h2_proj1)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj1, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          # h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          # s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          # h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_concatDepth_convAfterProjection_noZ3_smaller(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16 // 2, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8 // 2, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4 // 2, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16 // 2, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8 // 2, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4 // 2, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in2, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          h2_all = tf.concat([h2_rotated, h2], -2)

          h2_proj1 = conv3d(h2_all, self.gf_dim * 1, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu( h2_proj1)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj1, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          # h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          # s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          # h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_withBackgroundSmaller_concat_convAfterProjection_noZ3_mask(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2 // 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1 // 2, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in2, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          h2_all = tf.concat([h2_rotated, h2], -1)

          h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu( h2_proj1)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj1, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          # h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          # s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          # h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7_foreground = tf.nn.relu(deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7_foreground'))
          h7_mask = tf.sigmoid(deconv2d(h6, [batch_size, s_h * 2, s_w * 2, 1], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7_mask'))
          h7_background = tf.nn.relu(deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7_background'))

          return h7_foreground, h7_mask, h7_background

  def generator_AdaIN_res128_max_pool_convAfterProjection_noZ3(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)
      all_objects = []
      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)

              all_objects.append(h2_rotated)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 ], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2 , 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 ], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1 , 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in2, 16, 16)
              h2 = transform_voxel_to_match_image(h2)
              all_objects.append(h2)
          h2_all = self.objects_pool(all_objects)

          h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu( h2_proj1)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj1, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          # h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          # s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          # h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_withBackgroundSmaller_add_convAfterProjection_noZ3_mask(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 ], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2 , 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1 , 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in2, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          # h2_all = tf.concat([h2_rotated, h2], -1)
          h2_all = h2_rotated + h2

          h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu( h2_proj1)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj1, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          # h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          # s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          # h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7_foreground = tf.nn.relu(deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7_foreground'))
          h7_mask = tf.sigmoid(deconv2d(h6, [batch_size, s_h * 2, s_w * 2, 1], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7_mask'))
          h7_background = tf.nn.relu(deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7_background'))

          return h7_foreground, h7_mask, h7_background

  def generator_AdaIN_res128_withBackground_concat_convAfterProjection_attention(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8 // 2, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2 // 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim * 1 //2, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1')
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2 // 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3, name='g_h2')
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1 // 2, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in2, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          h2_all = tf.concat([h2_rotated, h2], -1)

          h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu( h2_proj1)

          h2_proj1 = self.attention_3d(h2_proj1, self.gf_dim * 1, "attention3D")

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj1, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3 = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3')
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_withBackground_add(self, z1, z2, z3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z1)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('foreground'):
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w_fg', [s_h16, s_w16, s_d16, self.gf_dim * 8 ], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z1, self.gf_dim * 8, 'g_z0')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z1, self.gf_dim * 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_fg')  # n_filter = 256
              s2_fg, b2_fg = self.z_mapping_function(z1, self.gf_dim, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg)

              h2_rotated_fg = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated_fg)
          with tf.variable_scope('background'):
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z2, self.gf_dim * 8, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0)

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z2, self.gf_dim * 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1)

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
              s2, b2 = self.z_mapping_function(z2, self.gf_dim * 1, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2)

              h2 = tf_rotation_resampling(h2, view_in2, 16, 16)
              h2 = transform_voxel_to_match_image(h2)

          h2_all = h2_rotated + h2
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_all, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z3, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z3, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z3, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_attention(self, z, view_in, reuse=False):
      # Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64  # Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          # A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8],
                                  initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0),
                               (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3,
                        name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3,
                        name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_rotated = self.attention_3d(h2_rotated, self.gf_dim, "attention_3d")
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1,
                                              d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4 = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4 = self.AdaIn(h4, s4, b4)
          h4 = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_lrelu(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1],  k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4,  name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4,  name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4,  name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_lrelu_res128_withBackground(self, z_1, z_2, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z_1)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          with tf.variable_scope("foreground") as scope:
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2],
                                      initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0),
                                   (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z_1, self.gf_dim * 8 // 2, 'g_z0_fg')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg, 'g_prelu0_fg')

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z_1, self.gf_dim * 2 // 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg, 'g_prelu1_fg')

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3,name='g_h2_fg')  # n_filter = 128
              s2_fg, b2_fg = self.z_mapping_function(z_1, self.gf_dim * 1 // 2, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg, 'g_prelu2_fg')

              h2_rotated = tf_rotation_resampling(h2_fg, view_in, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated)

          with tf.variable_scope("background") as scope:
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2],
                                      initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0),
                                   (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z_2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0, 'g_prelu0')

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3,
                            name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z_2, self.gf_dim * 2 // 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1, 'g_prelu1')

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3,name='g_h2')  # n_filter = 128
              s2, b2 = self.z_mapping_function(z_2, self.gf_dim * 1 // 2, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2, 'g_prelu2')
              h2_bg = transform_voxel_to_match_image(h2)
          #Concatenate foreground and bakground
          h2_all = tf.concat([h2_rotated, h2_bg], axis=-1)

          # h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          # h2_proj1 = lrelu( h2_proj1)
          #
          # h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          # h2_proj2 = lrelu( h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_all, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z_2, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z_2, self.gf_dim, 'g_z5')
          # h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          # s6, b6 = self.z_mapping_function(z_2, self.gf_dim // 2, 'g_z6')
          # h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_lrelu_res128_withBackground_adding(self, z_1, z_2, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z_1)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          with tf.variable_scope("foreground") as scope:
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2],
                                      initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0),
                                   (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z_1, self.gf_dim * 8 // 2, 'g_z0_fg')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg, 'g_prelu0_fg')

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z_1, self.gf_dim * 2 // 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg, 'g_prelu1_fg')

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3,name='g_h2_fg')  # n_filter = 128
              s2_fg, b2_fg = self.z_mapping_function(z_1, self.gf_dim * 1 // 2, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg, 'g_prelu2_fg')

              h2_rotated = tf_rotation_resampling(h2_fg, view_in, 16, 16)
              h2_rotated = transform_voxel_to_match_image(h2_rotated)

          with tf.variable_scope("background") as scope:
              with tf.variable_scope('g_w_constant'):
                  w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2],
                                      initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile = tf.tile(tf.expand_dims(w, 0),
                                   (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0, b0 = self.z_mapping_function(z_2, self.gf_dim * 8 // 2, 'g_z0')
                  h0 = self.AdaIn(w_tile, s0, b0)
                  h0 = tf.nn.relu(h0, 'g_prelu0')

              h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3,
                            name='g_h1')  # n_filter = 256
              s1, b1 = self.z_mapping_function(z_2, self.gf_dim * 2 // 2, 'g_z1')
              h1 = self.AdaIn(h1, s1, b1)
              h1 = tf.nn.relu(h1, 'g_prelu1')

              h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3,name='g_h2')  # n_filter = 128
              s2, b2 = self.z_mapping_function(z_2, self.gf_dim * 1 // 2, 'g_z2')
              h2 = self.AdaIn(h2, s2, b2)
              h2 = tf.nn.relu(h2, 'g_prelu2')

          h2_bg = transform_voxel_to_match_image(h2)
          #Concatenate foreground and bakground
          h2_all = h2_rotated + h2_bg

          # h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          # h2_proj1 = lrelu( h2_proj1)
          #
          # h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          # h2_proj2 = lrelu( h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_all, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim // 2])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z_2, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z_2, self.gf_dim, 'g_z5')
          # h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          # s6, b6 = self.z_mapping_function(z_2, self.gf_dim // 2, 'g_z6')
          # h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_relu_res128_withBackground_2Z(self, z_1, z_2, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z_1)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          with tf.variable_scope("foreground") as scope:
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2],
                                      initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0),
                                   (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z_1, self.gf_dim * 8 // 2, 'g_z0_fg')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = tf.nn.relu(h0_fg, 'g_prelu0_fg')

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z_1, self.gf_dim * 2 // 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = tf.nn.relu(h1_fg, 'g_prelu1_fg')

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3,name='g_h2_fg')  # n_filter = 128
              s2_fg, b2_fg = self.z_mapping_function(z_1, self.gf_dim * 1 // 2, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = tf.nn.relu(h2_fg, 'g_prelu2_fg')

              h2_fg_rotated = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_fg_rotated = transform_voxel_to_match_image(h2_fg_rotated)

          with tf.variable_scope("background") as scope:
              with tf.variable_scope('g_w_constant'):
                  w_bg = tf.get_variable('w_bg', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2],
                                      initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_bg = tf.tile(tf.expand_dims(w_bg, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0_bg, b0_bg = self.z_mapping_function(z_2, self.gf_dim * 8 // 2, 'g_z0_bg')
                  h0_bg = self.AdaIn(w_tile_bg, s0_bg, b0_bg)
                  h0_bg = tf.nn.relu(h0_bg, 'g_prelu0')

              h1_bg = deconv3d(h0_bg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1_bg')  # n_filter = 256
              s1_bg, b1_bg = self.z_mapping_function(z_2, self.gf_dim * 2 // 2, 'g_z1_bg')
              h1_bg = self.AdaIn(h1_bg, s1_bg, b1_bg)
              h1_bg = tf.nn.relu(h1_bg, 'g_prelu1_bg')

              h2_bg = deconv3d(h1_bg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3,name='g_h2_bg')  # n_filter = 128
              s2_bg, b2_bg = self.z_mapping_function(z_2, self.gf_dim * 1 // 2, 'g_z2_bg')
              h2_bg = self.AdaIn(h2_bg, s2_bg, b2_bg)
              h2_bg = tf.nn.relu(h2_bg, 'g_prelu2_bg')

              h2_bg_rotated = tf_rotation_resampling(h2_bg, view_in2, 16, 16)
              h2_bg_rotated = transform_voxel_to_match_image(h2_bg_rotated)

          #Concatenate foreground and bakground
          h2_all = tf.concat([h2_fg_rotated, h2_bg_rotated], axis=-1)

          # h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          # h2_proj1 = lrelu( h2_proj1)
          #
          h2_proj2 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=1, k_w=1, k_d=1, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = lrelu( h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z_2, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z_2, self.gf_dim, 'g_z5')
          # h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          # s6, b6 = self.z_mapping_function(z_2, self.gf_dim // 2, 'g_z6')
          # h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_lrelu_res128_withBackground_2Z_adding(self, z_1, z_2, z_3, view_in1, view_in2, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z_1)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          with tf.variable_scope("foreground") as scope:
              with tf.variable_scope('g_w_constant'):
                  w_fg = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2],
                                      initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_fg = tf.tile(tf.expand_dims(w_fg, 0),
                                   (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0_fg, b0_fg = self.z_mapping_function(z_1, self.gf_dim * 8 // 2, 'g_z0_fg')
                  h0_fg = self.AdaIn(w_tile_fg, s0_fg, b0_fg)
                  h0_fg = lrelu(h0_fg)

              h1_fg = deconv3d(h0_fg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1_fg')  # n_filter = 256
              s1_fg, b1_fg = self.z_mapping_function(z_1, self.gf_dim * 2 // 2, 'g_z1_fg')
              h1_fg = self.AdaIn(h1_fg, s1_fg, b1_fg)
              h1_fg = lrelu(h1_fg)

              h2_fg = deconv3d(h1_fg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3,name='g_h2_fg')  # n_filter = 128
              s2_fg, b2_fg = self.z_mapping_function(z_1, self.gf_dim * 1 // 2, 'g_z2_fg')
              h2_fg = self.AdaIn(h2_fg, s2_fg, b2_fg)
              h2_fg = lrelu(h2_fg)

              h2_fg_rotated = tf_rotation_resampling(h2_fg, view_in1, 16, 16)
              h2_fg_rotated = transform_voxel_to_match_image(h2_fg_rotated)

          with tf.variable_scope("background") as scope:
              with tf.variable_scope('g_w_constant'):
                  w_bg = tf.get_variable('w_bg', [s_h16, s_w16, s_d16, self.gf_dim * 8 // 2],
                                      initializer=tf.random_normal_initializer(stddev=0.02))
                  w_tile_bg = tf.tile(tf.expand_dims(w_bg, 0), (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
                  s0_bg, b0_bg = self.z_mapping_function(z_2, self.gf_dim * 8 // 2, 'g_z0_bg')
                  h0_bg = self.AdaIn(w_tile_bg, s0_bg, b0_bg)
                  h0_bg = lrelu(h0_bg)

              h1_bg = deconv3d(h0_bg, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2 // 2], k_h=3, k_w=3, k_d=3, name='g_h1_bg')  # n_filter = 256
              s1_bg, b1_bg = self.z_mapping_function(z_2, self.gf_dim * 2 // 2, 'g_z1_bg')
              h1_bg = self.AdaIn(h1_bg, s1_bg, b1_bg)
              h1_bg = lrelu(h1_bg)

              h2_bg = deconv3d(h1_bg, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1 // 2], k_h=3, k_w=3, k_d=3,name='g_h2_bg')  # n_filter = 128
              s2_bg, b2_bg = self.z_mapping_function(z_2, self.gf_dim * 1 // 2, 'g_z2_bg')
              h2_bg = self.AdaIn(h2_bg, s2_bg, b2_bg)
              h2_bg = lrelu(h2_bg)

              h2_bg_rotated = tf_rotation_resampling(h2_bg, view_in2, 16, 16)
              h2_bg_rotated = transform_voxel_to_match_image(h2_bg_rotated)

          #Concatenate foreground and bakground
          h2_all = h2_fg_rotated + h2_bg_rotated

          # h2_proj1 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          # h2_proj1 = lrelu( h2_proj1)
          #
          h2_proj2 = deconv3d(h2_all, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = lrelu( h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z_3, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z_3, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z_3, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_res128_prelu(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      s_h, s_w, s_d = 64, 64, 64 #Hack to quickly extend baseline model to res 128x128 without all size and channels suddenly double up
      batch_size = tf.shape(z)[0]
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = prelu(h0, 'g_prelu0')

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = prelu(h1, 'g_prelu1')

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = prelu(h2, 'g_prelu2')

          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = prelu(h3, 'g_prelu3')

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = prelu(h4, 'g_prelu4')

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = prelu(h5, 'g_prelu5')

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = prelu(h6, 'g_prelu6')

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convBeforeProjection(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          s2_proj1, b2_proj1 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_proj1')
          h2_proj1 = self.AdaIn(h2_proj1, s2_proj1, b2_proj1)
          h2_proj1 = tf.nn.relu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          s2_proj2, b2_proj2 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z2_proj2')
          h2_proj2 = self.AdaIn(h2_proj2, s2_proj2, b2_proj2)
          h2_proj2 = tf.nn.relu(h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, 16 * self.gf_dim // 2 ])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          # s2_proj1, b2_proj1 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_proj1')
          # h2_proj1 = self.AdaIn(h2_proj1, s2_proj1, b2_proj1)
          h2_proj1 = tf.nn.relu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          # s2_proj2, b2_proj2 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z2_proj2')
          # h2_proj2 = self.AdaIn(h2_proj2, s2_proj2, b2_proj2)
          h2_proj2 = tf.nn.relu(h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim],  k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_res128(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          # s2_proj1, b2_proj1 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_proj1')
          # h2_proj1 = self.AdaIn(h2_proj1, s2_proj1, b2_proj1)
          h2_proj1 = tf.nn.relu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          # s2_proj2, b2_proj2 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z2_proj2')
          # h2_proj2 = self.AdaIn(h2_proj2, s2_proj2, b2_proj2)
          h2_proj2 = tf.nn.relu(h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = tf.nn.relu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_lrelu(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          # s2_proj1, b2_proj1 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_proj1')
          # h2_proj1 = self.AdaIn(h2_proj1, s2_proj1, b2_proj1)
          h2_proj1 = lrelu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          # s2_proj2, b2_proj2 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z2_proj2')
          # h2_proj2 = self.AdaIn(h2_proj2, s2_proj2, b2_proj2)
          h2_proj2 = lrelu(h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_lrelu_res128(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2],  k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = lrelu( h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = lrelu( h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_lrelu_res128_specNorm(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1= deconv3d_specNorm(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d_specNorm(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2],  k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d_specNorm(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = lrelu( h2_proj1)

          h2_proj2 = deconv3d_specNorm(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = lrelu( h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d_specNorm(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          h5 = deconv2d_specNorm(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d_specNorm(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_lrelu_res128_2(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0', act='lrelu')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1', act='lrelu')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2],  k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2', act='lrelu')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = lrelu( h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = lrelu( h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4', act='lrelu')
          h4  = self.AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5', act='lrelu')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6', act='lrelu')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_lrelu_res128_2_specNorm(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0', act='lrelu')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1= deconv3d_specNorm(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1', act='lrelu')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d_specNorm(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2],  k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2', act='lrelu')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d_specNorm(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = lrelu( h2_proj1)

          h2_proj2 = deconv3d_specNorm(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = lrelu( h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d_specNorm(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4', act='lrelu')
          h4  = self.AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          h5 = deconv2d_specNorm(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5', act='lrelu')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d_specNorm(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6', act='lrelu')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_lrelu_res128_3(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0, leak=0.1)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1, leak=0.1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2],  k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2, leak=0.1)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = lrelu( h2_proj1, leak=0.1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = lrelu( h2_proj2, leak=0.1)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3, leak=0.1)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4 = lrelu(h4, leak=0.1)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5, leak=0.1)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6, leak=0.1)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_lrelu_res128_multiZ(self, z, z2, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2],  k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = lrelu( h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = lrelu( h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = lrelu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z2, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z2, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z2, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_prelu(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = prelu(h0, 'g_prelu0')

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = prelu(h1, 'g_prelu1')

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = prelu(h2, 'g_prelu2')

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = prelu( h2_proj1, 'g_prelu_h2_proj1')

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = prelu( h2_proj2, 'g_prelu_h2_proj2')

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = prelu(h3, 'g_prelu3')

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4 = prelu(h4, 'g_prelu4')

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4,  name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = prelu(h5, 'g_prelu5')

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4,  d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_prelu_res128(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = prelu(h0, 'g_prelu0')

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_d=3, k_w=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = prelu(h1, 'g_prelu1')

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], k_h=3, k_d=3, k_w=3,  name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = prelu(h2, 'g_prelu2')

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = prelu( h2_proj1, 'g_prelu_h2_proj1')

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = prelu( h2_proj2, 'g_prelu_h2_proj2')

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = prelu(h3, 'g_prelu3')

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4 = prelu(h4, 'g_prelu4')

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = prelu(h5, 'g_prelu5')

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = prelu(h6, 'g_prelu6')

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_prelu_res128_2(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      # s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0', act='prelu')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = prelu(h0, 'g_prelu0')

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_d=3, k_w=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1', act='prelu')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = prelu(h1, 'g_prelu1')

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], k_h=3, k_d=3, k_w=3,  name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2', act='prelu')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = prelu(h2, 'g_prelu2')

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = prelu( h2_proj1, 'g_prelu_h2_proj1')

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = prelu( h2_proj2, 'g_prelu_h2_proj2')

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = prelu(h3, 'g_prelu3')

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4', act='prelu')
          h4  = self.AdaIn(h4, s4, b4)
          h4 = prelu(h4, 'g_prelu4')

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5', act='prelu')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = prelu(h5, 'g_prelu5')

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6', act='prelu')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = prelu(h6, 'g_prelu6')

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_moreSpatialConv3D_convWithoutINBeforeProjection_lrelu(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          h3 = deconv3d(h2, [batch_size, s_h2, s_w2, s_d2, self.gf_dim], k_h=3, k_w=3, k_d=3, name='g_h3')  # n_filter = 128
          s3, b3 = self.z_mapping_function(z, self.gf_dim, 'g_z3')
          h3 = self.AdaIn(h3, s3, b3)
          h3 = lrelu(h3)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h3, view_in, 32, 32)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h2, s_w2, s_d2, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = lrelu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h2, s_w2, s_d2, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = lrelu(h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h2, s_w2, s_d2 * self.gf_dim // 2])
          # 1X1 convolution
          h4 = deconv2d(h2_2d, [batch_size, s_h2, s_w2, self.gf_dim * s_d2 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h4')
          h4 = lrelu(h4)

          # h4  = deconv2d(h4, [batch_size, s_h2, s_w2, self.gf_dim * 4], name='g_h4')
          # s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          # h4 = prelu(h4, 'g_prelu4')

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_moreSpatialConv3D_convWithoutINBeforeProjection_lrelu_res128(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4],  k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2],  k_h=3, k_w=3, k_d=3,name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          h3 = deconv3d(h2, [batch_size, s_h2, s_w2, s_d2, self.gf_dim],  k_h=3, k_w=3, k_d=3, name='g_h3')  # n_filter = 128
          s3, b3 = self.z_mapping_function(z, self.gf_dim, 'g_z3')
          h3 = self.AdaIn(h3, s3, b3)
          h3 = lrelu(h3)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h3, view_in, 32, 32)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h2, s_w2, s_d2, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = lrelu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h2, s_w2, s_d2, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = lrelu(h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h2, s_w2, s_d2 * self.gf_dim // 2])
          # 1X1 convolution
          h4 = deconv2d(h2_2d, [batch_size, s_h2, s_w2, self.gf_dim * s_d2 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h4')
          h4 = lrelu(h4)

          h5  = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim * 4], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z5')
          h5  = self.AdaIn(h5, s5, b5)
          h5 = prelu(h5, 'g_prelu5')

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w*2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_moreSpatialConv3D_convWithoutINBeforeProjection_lrelu_res128_multiZ(self, z, z2, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4],  k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2],  k_h=3, k_w=3, k_d=3,name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          h3 = deconv3d(h2, [batch_size, s_h2, s_w2, s_d2, self.gf_dim],  k_h=3, k_w=3, k_d=3, name='g_h3')  # n_filter = 128
          s3, b3 = self.z_mapping_function(z, self.gf_dim, 'g_z3')
          h3 = self.AdaIn(h3, s3, b3)
          h3 = lrelu(h3)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h3, view_in, 32, 32)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h2, s_w2, s_d2, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = lrelu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h2, s_w2, s_d2, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = lrelu(h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h2, s_w2, s_d2 * self.gf_dim // 2])
          # 1X1 convolution
          h4 = deconv2d(h2_2d, [batch_size, s_h2, s_w2, self.gf_dim * s_d2 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h4')
          h4 = lrelu(h4)

          h5  = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim * 4], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z2, self.gf_dim * 4, 'g_z5')
          h5  = self.AdaIn(h5, s5, b5)
          h5 = prelu(h5, 'g_prelu5')

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z2, self.gf_dim, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w*2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_moreSpatialConv3D_convWithoutINBeforeProjection_prelu(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = prelu(h0, 'g_prelu0')

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = prelu(h1, 'g_prelu1')

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = prelu(h2, 'g_prelu2')

          h3 = deconv3d(h2, [batch_size, s_h2, s_w2, s_d2, self.gf_dim], k_h=3, k_w=3, k_d=3, name='g_h3')  # n_filter = 128
          s3, b3 = self.z_mapping_function(z, self.gf_dim, 'g_z3')
          h3 = self.AdaIn(h3, s3, b3)
          h3 = prelu(h3, 'g_prelu3')

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h3, view_in, 32, 32)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h2, s_w2, s_d2, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = prelu( h2_proj1, 'g_prelu_h2_proj1')

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h2, s_w2, s_d2, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = prelu( h2_proj2, 'g_prelu_h2_proj2')

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h2, s_w2, s_d2 * self.gf_dim // 2])
          # 1X1 convolution
          h4 = deconv2d(h2_2d, [batch_size, s_h2, s_w2, self.gf_dim * s_d2 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h4')
          h4 = prelu(h4, 'g_prelu4')

          # h4  = deconv2d(h4, [batch_size, s_h2, s_w2, self.gf_dim * 4], name='g_h4')
          # s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          # h4  = self.AdaIn(h4, s4, b4)
          # h4 = prelu(h4, 'g_prelu4')

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim],  k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = prelu(h5, 'g_prelu5')

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_moreSpatialConv3D_convWithoutINBeforeProjection_prelu_res128(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = prelu(h0, 'g_prelu0')

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = prelu(h1, 'g_prelu1')

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = prelu(h2, 'g_prelu2')

          h3 = deconv3d(h2, [batch_size, s_h2, s_w2, s_d2, self.gf_dim], k_h=3, k_w=3, k_d=3, name='g_h3')  # n_filter = 128
          s3, b3 = self.z_mapping_function(z, self.gf_dim, 'g_z3')
          h3 = self.AdaIn(h3, s3, b3)
          h3 = prelu(h3, 'g_prelu3')

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h3, view_in, 32, 32)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h2, s_w2, s_d2, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = prelu( h2_proj1, 'g_prelu_h2_proj1')

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h2, s_w2, s_d2, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = prelu( h2_proj2, 'g_prelu_h2_proj2')

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h2, s_w2, s_d2 * self.gf_dim // 2])
          # 1X1 convolution
          h4 = deconv2d(h2_2d, [batch_size, s_h2, s_w2, self.gf_dim * s_d2 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h4')
          h4 = prelu(h4, 'g_prelu4')

          h5  = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim * 4], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z5')
          h5  = self.AdaIn(h5, s5, b5)
          h5 = prelu(h5, 'g_prelu5')

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim, 'g_z6')
          h6 = self.AdaIn(h6, s6, b6)
          h6 = prelu(h6, 'g_prelu6')

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          return output

  def generator_AdaIN_1convWithoutINBeforeProjection(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)


          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)


          # =============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1,
                              d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu(h2_proj1)

          # h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1,
          #                     d_d=1, name='g_h2_proj2')  # n_filter = 64
          # # s2_proj2, b2_proj2 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z2_proj2')
          # # h2_proj2 = self.AdaIn(h2_proj2, s2_proj2, b2_proj2)
          # h2_proj2 = tf.nn.relu(h2_proj2)

          # =============================================================================================================

          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj1, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_changeConvSize(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          h2_proj1 = tf.nn.relu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim // 2], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          h2_proj2 = tf.nn.relu(h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim //2])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim  // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_1x1convWithoutINBeforeProjection_changeConvSize(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          # s2_proj1, b2_proj1 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_proj1')
          # h2_proj1 = self.AdaIn(h2_proj1, s2_proj1, b2_proj1)
          h2_proj1 = tf.nn.relu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim // 2], k_h=1, k_w=1, k_d=1, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          # s2_proj2, b2_proj2 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z2_proj2')
          # h2_proj2 = self.AdaIn(h2_proj2, s2_proj2, b2_proj2)
          h2_proj2 = tf.nn.relu(h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim // 2])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * s_d4 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_convWithoutINBeforeProjection_biggerSpatialConv3D(self, z, view_in, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          h2_2 = deconv3d(h2, [batch_size, s_h2, s_w2, s_d2, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, name='g_h2_2')  # n_filter = 128
          s2_2, b2_2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_2')
          h2_2 = self.AdaIn(h2_2, s2_2, b2_2)
          h2_2 = tf.nn.relu(h2_2)

          #=============================================================================================================
          h2_rotated = tf_rotation_resampling(h2_2, view_in, 32, 32)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h2, s_w2, s_d2, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')  # n_filter = 64
          # s2_proj1, b2_proj1 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2_proj1')
          # h2_proj1 = self.AdaIn(h2_proj1, s2_proj1, b2_proj1)
          h2_proj1 = tf.nn.relu(h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h2, s_w2, s_d2, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')  # n_filter = 64
          # s2_proj2, b2_proj2 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z2_proj2')
          # h2_proj2 = self.AdaIn(h2_proj2, s2_proj2, b2_proj2)
          h2_proj2 = tf.nn.relu(h2_proj2)

          # =============================================================================================================
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h2, s_w2, s_d2 * self.gf_dim])
          # 1X1 convolution
          h3, self.h3_w, self.h3_b = deconv2d(h2_2d, [batch_size, s_h2, s_w2, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3', with_w=True)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], d_h=1, d_w=1, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = self.AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = self.AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_2D(self, z, reuse=False):
      #Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = self.output_height, self.output_width, self.output_width  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, self.gf_dim * 8], initializer = tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = self.AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv2d(h0, [batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = self.AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv2d(h1, [batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')  # n_filter = 128
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = self.AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          h3 = deconv2d(h2, [batch_size, s_h2, s_w2, self.gf_dim], name='g_h3')  # n_filter = 64
          s3, b3 = self.z_mapping_function(z, self.gf_dim, 'g_z3')
          h3 = self.AdaIn(h3, s3, b3)
          h3 = tf.nn.relu(h3)

          h4  = deconv2d(h3, [batch_size, s_h, s_w, self.c_dim], name='g_h4')


          output = tf.nn.tanh(h4, name="output")
          return output

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0


