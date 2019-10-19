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




def sampling_Z(batch_size, z_dim, type="uniform"):
    if str.lower(type) == "uniform":
        return np.random.uniform(-1., 1., (batch_size, z_dim))
    else:
        return np.random.normal(0, 1, (batch_size, z_dim))

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

#===========================================================================================================
#Activation functions
#===========================================================================================================
def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

#===========================================================================================================
#Normalization
#===========================================================================================================
def AdaIn(features, scale, bias, name="AdaIn"):
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

def instance_norm(input, name="instance_norm"):
  """
  Taken from https://github.com/xhujoy/CycleGAN-tensorflow/blob/master/module.py
  :param input:
  :param name:
  :return:
  """
  with tf.variable_scope(name):
      depth = input.get_shape()[-1]
      scale = tf.get_variable("scale", [depth],
                              initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
      offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
      mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
      epsilon = 1e-5
      inv = tf.rsqrt(variance + epsilon)
      normalized = (input - mean) * inv
      return scale * normalized + offset

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

def z_mapping_function(z, output_channel, scope='z_mapping', act="relu", stddev=0.02):
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

#===========================================================================================================
#Convolutions
#===========================================================================================================
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
  print(shape)
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("weights", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("biases", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def linear_fixed_size(input_, input_size, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()
  print(shape)
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("weights", [input_size, output_size], tf.float32,
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



if __name__ == "__main__":
    import binvox_rw

    path = r"D:\Data_sets\ShapeNetCore_v2\ShapeNet64_Chair\ShapeNet64_Chair_binvox_2_centered\model_chair_13_clean.binvox"
    with open(path, "rb") as fp:
        mod = binvox_rw.read_as_3d_array(fp).data.astype(np.float32)
        mod = np.expand_dims(np.expand_dims(mod, 0), 4)
    vox_in = tf.placeholder(tf.float32, (1, 64, 64, 64, 1))
    result = upsample(vox_in, 4, 1)
    with tf.Session() as sess:
        a = sess.run(result, feed_dict={vox_in: mod})
        binvox_rw.save_binvox(np.squeeze(a) > 0.5, r"D:/up4.binvox")