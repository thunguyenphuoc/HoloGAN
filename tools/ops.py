from .utils import *
import tensorflow as tf
import numpy as np
try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter

def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

#===========================================================================================================
#Activation functions
#===========================================================================================================

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

#===========================================================================================================
#Normalization
#===========================================================================================================
def AdaIn(features, scale, bias):
    """
    Adaptive instance normalization component. Works with both 4D and 5D tensors
    :features: features to be normalized
    :scale: scaling factor. This would otherwise be calculated as the sigma from a "style" features in style transfer
    :bias: bias factor. This would otherwise be calculated as the mean from a "style" features in style transfer
    """

    mean, variance = tf.nn.moments(features, list(range(len(features.get_shape())))[1:-1],
                                   keep_dims=True)  # Only consider spatial dimension
    sigma = tf.rsqrt(variance + 1e-8)
    normalized = (features - mean) * sigma
    scale_broadcast = tf.reshape(scale, tf.shape(mean))
    bias_broadcast = tf.reshape(bias, tf.shape(mean))
    normalized = scale_broadcast * normalized
    normalized += bias_broadcast
    return normalized

def instance_norm(input, name="instance_norm", return_mean=False):
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

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_norm(w, iteration=1, u_weight=None):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    if u_weight is None:
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    else:
        u = u_weight

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


#===========================================================================================================
#Convolutions
#===========================================================================================================
def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None:
        fan_in = np.prod(shape[:-1])
    print ("current", shape[:-1], fan_in)
    std = gain / np.sqrt(fan_in) # He init

    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))


def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d", padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv


def conv2d_specNorm(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2dSpectral", padding='SAME'):
  with tf.variable_scope(name):
    w = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    return conv

def conv3d(input_, output_dim,
       k_h=5, k_w=5, k_d=5, d_h=2, d_w=2, d_d=2, stddev=0.02,
       name="conv3d", padding='SAME'):
  with tf.variable_scope(name):
    w = tf.get_variable('weights', [k_h, k_w, k_d, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, d_d, 1], padding=padding)

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    return conv

def conv3d_specNorm(input_, output_dim,
       k_h=5, k_w=5, k_d=5, d_h=2, d_w=2, d_d=2, stddev=0.02,
       name="conv3dSpectral", padding='SAME'):
  with tf.variable_scope(name):
    w = tf.get_variable('weights', [k_h, k_w, k_d, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv3d(input_, spectral_norm(w), strides=[1, d_h, d_w, d_d, 1], padding=padding)

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('weights', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))


    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf.shape(deconv))

    if with_w:
      return deconv, w, biases
    else:
      return deconv


def deconv2d_specNorm(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('weights', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, spectral_norm(w), output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf.shape(deconv))

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def deconv3d(input_, output_shape,
             k_h=5, k_w=5, k_d=5, d_h=2, d_w=2, d_d=2, stddev=0.02,
             name="deconv3d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('weights', [k_h, k_w, k_d, output_shape[-1], input_.get_shape()[-1]],
                        initializer=tf.random_normal_initializer(stddev=stddev))

    deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape,
                                      strides=[1, d_h, d_w, d_d, 1])


    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf.shape(deconv))

    if with_w:
      return deconv, w, biases
    else:
      return deconv

def deconv3d_specNorm(input_, output_shape,
             k_h=5, k_w=5, k_d=5, d_h=2, d_w=2, d_d=2, stddev=0.02,
             name="deconv3dSpectral", with_w=False):
  with tf.variable_scope(name):
    w = tf.get_variable('weights', [k_h, k_w, k_d, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv3d_transpose(input_, spectral_norm(w), output_shape=output_shape, strides=[1, d_h, d_w, d_d, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf.shape(deconv))

    if with_w:
      return deconv, w, biases
    else:
      return deconv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("weights", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("biases", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def linear_specNorm(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = spectral_norm(tf.get_variable("weights", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev)))
    bias = tf.get_variable("biases", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias


def flatten(x) :
    return tf.layers.flatten(x)


