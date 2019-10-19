import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib
# from tools.ops import *




def get_weight(weight_name, weight_dict):
    if weight_dict is None:
        return None
    else:
        return weight_dict.get(weight_name)  # returns None if name is not found in dictionary

def res_block_3d(input, out_channels=64, scope = 'res_block', kernel=[3, 3, 3], prob = 0.5,  stride=[1, 1, 1], weight_dict=None):
    with tf.variable_scope(scope):
        net = tf.nn.relu(conv3d(input, out_channels, kernel_size=kernel, stride=stride, pad="SAME", scope="con1_3X3",
                                weight_initializer=get_weight(scope + 'con1_3X3_weights', weight_dict),
                                bias_initializer=get_weight(scope + 'con1_3X3_biases', weight_dict),
                                weight_initializer_type=tf.contrib.layers.xavier_initializer()))
        # net = tf.nn.dropout(net, keep_prob(prob, is_training))
        net = conv3d(net, out_channels, kernel_size=kernel, stride=stride, pad="SAME", scope="conv2_3x3",
                     weight_initializer=get_weight(scope + 'conv2_3x3_weights', weight_dict),
                     bias_initializer=get_weight(scope + 'conv2_3x3_biases', weight_dict),
                     weight_initializer_type=tf.contrib.layers.xavier_initializer())
        # net = tf.nn.dropout(net, keep_prob(prob, is_training))
    return tf.add(tf.cast(net, tf.float32), tf.cast(input, tf.float32))


def res_block_2d(input, out_channels=64, scope = 'res_block', kernel=[3, 3], prob = 0.5,  stride=[1, 1], weight_dict=None):
    with tf.variable_scope(scope):
        net = tf.nn.relu(conv2d(input, out_channels, kernel_size=kernel, stride=stride, pad="SAME", scope="con1_3X3",
                                weight_initializer=get_weight(scope + 'con1_3X3_weights', weight_dict),
                                bias_initializer=get_weight(scope + 'con1_3X3_biases', weight_dict),
                                weight_initializer_type=tf.contrib.layers.xavier_initializer()))
        # net = tf.nn.dropout(net, keep_prob(prob, is_training))
        net = conv2d(net, out_channels, kernel_size=kernel, stride=stride, pad="SAME", scope="conv2_3x3",
                     weight_initializer=get_weight(scope + 'conv2_3x3_weights', weight_dict),
                     bias_initializer=get_weight(scope + 'conv2_3x3_biases', weight_dict),
                     weight_initializer_type=tf.contrib.layers.xavier_initializer())
        # net = tf.nn.dropout(net, keep_prob(prob, is_training))
    return tf.add(tf.cast(net, tf.float32), tf.cast(input, tf.float32))


def bias_variable(shape, bias_initializer=None, trainable=True):
    if bias_initializer is None :
        return tf.get_variable(name='biases', initializer=tf.constant(0.0, shape=shape), trainable = trainable)
    else:
        return tf.get_variable(name='biases', initializer=bias_initializer, trainable = trainable)


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1, seed=1), dtype=tf.float32)
    return weights_init

def conv2d(input_, num_outputs, kernel_size=[4,4], stride=[1,1], pad='SAME', if_bias=True, trainable=True, reuse=False,
           scope='conv2d', weight_initializer=None, bias_initializer=None,
           weight_initializer_type=tf.random_normal_initializer(stddev=0.02)):
    print(scope)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initializing weights")
            w = tf.get_variable(name='weights',
                                shape = kernel_size +  [input_.get_shape()[-1]] + [num_outputs],
                                initializer = weight_initializer_type,
                                dtype = tf.float32, trainable=trainable)
        else:
            print("Loading weights")
            w = tf.get_variable(name='weights',
                                initializer = weight_initializer,
                                dtype = tf.float32, trainable=trainable)

        conv = tf.nn.conv2d(input_, w,
                            padding = pad,
                            strides = [1] + stride + [1])

        if if_bias:
            if bias_initializer is None:
                print("Initializing biases")
                conv=conv+bias_variable([num_outputs], trainable=trainable)
            else:
                print("Loading biases")
                conv=conv+bias_variable([num_outputs], trainable=trainable, bias_initializer=bias_initializer)

        return conv


def conv2d_transpose(x, num_outputs, kernel_size = (4,4), stride= (1,1), pad='SAME', if_bias=True,
                     reuse=False, scope = "conv2d_transpose", trainable = True, weight_initializer=None,
                     bias_initializer=None, weight_initializer_type=tf.random_normal_initializer(stddev=0.02)):
    print(scope)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initializing weights")
            w = tf.get_variable(name='weights',
                                shape = kernel_size + [num_outputs] + [x.get_shape()[-1]],
                                initializer=weight_initializer_type,
                                dtype = tf.float32, trainable=trainable)
        else:
            print("Loading weights")
            w = tf.get_variable(name='weights',
                                initializer=weight_initializer,
                                dtype = tf.float32, trainable=trainable)

        output_shape = [tf.shape(x)[0], tf.shape(x)[1] * stride[0], tf.shape(x)[2] * stride[1], num_outputs]

        conv_trans = tf.nn.conv2d_transpose(x, w,
                                            output_shape = output_shape,
                                            strides = [1] + stride + [1],
                                            padding = pad)

        if if_bias:
            if bias_initializer is None:
                print("Initializing biases")
                conv_trans = conv_trans + bias_variable([num_outputs], trainable=trainable)
            else:
                print("Load biases")
                conv_trans = conv_trans + bias_variable([num_outputs], trainable=trainable, bias_initializer=bias_initializer)


        return conv_trans

def conv2d_specnorm(input_, num_outputs, kernel_size=[4,4], stride=[1,1], pad='SAME', if_bias=True, trainable=True, reuse=False,
           scope='conv2d', weight_initializer=None, bias_initializer=None, u_weight=None,
           weight_initializer_type=tf.random_normal_initializer(stddev=0.02)):
    print(scope)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initializing weights")
            w = tf.get_variable(name='weights',
                                shape = kernel_size +  [input_.get_shape()[-1]] + [num_outputs],
                                initializer = weight_initializer_type,
                                dtype = tf.float32, trainable=trainable)
        else:
            print("Loading weights")
            w = tf.get_variable(name='weights',
                                initializer = weight_initializer,
                                dtype = tf.float32, trainable=trainable)

        conv = tf.nn.conv2d(input_, spectral_norm(w, 1, u_weight=u_weight),
                            padding = pad,
                            strides = [1] + stride + [1])

        if if_bias:
            if bias_initializer is None:
                print("Initializing biases")
                conv=conv+bias_variable([num_outputs], trainable=trainable)
            else:
                print("Loading biases")
                conv=conv+bias_variable([num_outputs], trainable=trainable, bias_initializer=bias_initializer)

        return conv

def conv2d_transpose_specNorm(x, num_outputs, kernel_size = (4,4), stride= (1,1), pad='SAME', if_bias=True,
                     reuse=False, scope = "conv2d_transpose", trainable = True, weight_initializer=None,
                     bias_initializer=None, u_weight=None, weight_initializer_type=tf.random_normal_initializer(stddev=0.02)):
    print(scope)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initializing weights")
            w = tf.get_variable(name='weights',
                                shape = kernel_size + [num_outputs] + [x.get_shape()[-1]],
                                initializer=weight_initializer_type,
                                dtype = tf.float32, trainable=trainable)
        else:
            print("Loading weights")
            w = tf.get_variable(name='weights',
                                initializer=weight_initializer,
                                dtype = tf.float32, trainable=trainable)

        output_shape = [tf.shape(x)[0], tf.shape(x)[1] * stride[0], tf.shape(x)[2] * stride[1], num_outputs]

        conv_trans = tf.nn.conv2d_transpose(x, spectral_norm(w, 1, u_weight),
                                            output_shape = output_shape,
                                            strides = [1] + stride + [1],
                                            padding = pad)

        if if_bias:
            if bias_initializer is None:
                print("Initializing biases")
                conv_trans = conv_trans + bias_variable([num_outputs], trainable=trainable)
            else:
                print("Load biases")
                conv_trans = conv_trans + bias_variable([num_outputs], trainable=trainable, bias_initializer=bias_initializer)


        return conv_trans

def conv3d(input_, num_outputs, pad = "SAME", reuse = False, kernel_size = [4,4,4], stride = [2,2,2], if_bias= True,
           trainable = True, scope = "conv3d", weight_initializer=None, bias_initializer=None, weight_initializer_type = tf.random_normal_initializer(stddev=0.02)):
    print(scope)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initialise weight")
            w = tf.get_variable(name='weights',
                                trainable=trainable,
                                shape=kernel_size + [input_.get_shape()[-1]] + [num_outputs],
                                initializer=weight_initializer_type,
                                dtype=tf.float32)
        else:
            print("Loading weight")
            w = tf.get_variable(name='weights',
                                trainable=trainable,
                                initializer=weight_initializer,
                                dtype=tf.float32)

        conv = tf.nn.conv3d(input_, w,
                            padding = pad,
                            strides = [1] + stride + [1])

        if if_bias:
            if bias_initializer is None:
                print("Initialise bias")
                conv = conv + bias_variable([num_outputs], trainable=trainable)
            else:
                print("Loading bias")
                conv = conv + bias_variable([num_outputs], trainable=trainable, bias_initializer=bias_initializer)

        return conv



def conv3d_transpose(x, num_output, kernel_size = (4,4), stride= (1,1), pad='SAME', if_bias=True,
                     reuse=False, scope = "conv3d_transpose", trainable = True, weight_initializer=None,
                     bias_initializer=None, weight_initializer_type=tf.random_normal_initializer(stddev=0.02)):

    print(scope)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initializing weights")
            w = tf.get_variable(name='weights',
                                shape = kernel_size + [num_output] + [x.get_shape().as_list()[-1]],
                                initializer=weight_initializer_type,
                                dtype = tf.float32, trainable=trainable)
        else:
            print("Loading weights")
            w = tf.get_variable(name='weights',
                                initializer=weight_initializer,
                                dtype = tf.float32, trainable=trainable)
        print("W " + str(w.get_shape()))
        output_shape = [tf.shape(x)[0], tf.shape(x)[1] * stride[0], tf.shape(x)[2] * stride[1], tf.shape(x)[3] * stride[2], num_output]

        conv_trans = tf.nn.conv3d_transpose(x, w,
                                            output_shape = output_shape,
                                            strides = [1] + stride + [1],
                                            padding = pad)

        if if_bias:
            if bias_initializer is None:
                print("Initializing biases")
                conv_trans = conv_trans + bias_variable([num_output], trainable=trainable)
            else:
                print("Load biases")
                conv_trans = conv_trans + bias_variable([num_output], trainable=trainable, bias_initializer=bias_initializer)


        return conv_trans

def conv3d_transpose_specNorm(x, num_output, kernel_size = (4,4), stride= (1,1), pad='SAME', if_bias=True,
                     reuse=False, scope = "conv3d_transpose", trainable = True, weight_initializer=None,
                     bias_initializer=None, u_weight=None, weight_initializer_type=tf.random_normal_initializer(stddev=0.02)):

    print(scope)
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initializing weights")
            w = tf.get_variable(name='weights',
                                shape = kernel_size + [num_output] + [x.get_shape().as_list()[-1]],
                                initializer=weight_initializer_type,
                                dtype = tf.float32, trainable=trainable)
        else:
            print("Loading weights")
            w = tf.get_variable(name='weights',
                                initializer=weight_initializer,
                                dtype = tf.float32, trainable=trainable)
        print("W " + str(w.get_shape()))
        output_shape = [tf.shape(x)[0], tf.shape(x)[1] * stride[0], tf.shape(x)[2] * stride[1], tf.shape(x)[3] * stride[2], num_output]

        conv_trans = tf.nn.conv3d_transpose(x, spectral_norm(w, 1, u_weight),
                                            output_shape = output_shape,
                                            strides = [1] + stride + [1],
                                            padding = pad)

        if if_bias:
            if bias_initializer is None:
                print("Initializing biases")
                conv_trans = conv_trans + bias_variable([num_output], trainable=trainable)
            else:
                print("Load biases")
                conv_trans = conv_trans + bias_variable([num_output], trainable=trainable, bias_initializer=bias_initializer)


        return conv_trans

def fully_connected(input_, output_size, scope = 'fully_connected', if_bias=True,
                    weight_initializer=None, bias_initializer=None,  reuse = False, trainable = True,
                    weight_initializer_type=tf.random_normal_initializer(stddev=0.02)):
    print(scope)
    if (type(input_)== np.ndarray):
        shape = input_.shape
    else:
        shape = input_.get_shape().as_list()
        # shape = tf.shape(input_).value.as_list()
    with tf.variable_scope(scope, reuse = reuse):
        if weight_initializer is None:
            print("Initializing weights")
            matrix = tf.get_variable("weights", [shape[-1], output_size], initializer=weight_initializer_type, dtype = tf.float32, trainable=trainable)
        else:
            print("Loading weights")
            matrix = tf.get_variable("weights", initializer=weight_initializer, dtype=tf.float32, trainable=trainable)

        fc = tf.matmul(input_, matrix)
        if if_bias:
            if bias_initializer is None:
                print("Initializing biases")
                fc = fc + bias_variable([output_size], trainable=trainable)
            else:
                print("Load biases")
                fc = fc + bias_variable([output_size], bias_initializer, trainable=trainable)
        return fc

if __name__ == "__main__":
    import binvox_rw
    from binvox_checker import  visualise_voxel
    import tensorflow as tf

    with open(r"D:\Data_sets\ShapeNetCore_v2\ShapeNet64_Chair\ShapeNet64_Chair_binvox_3_centered\model_normalized_25_clean.binvox",
            'rb') as fp:
        shape_in_1 = binvox_rw.read_as_3d_array(fp).data.reshape([1,64,64,64,1])

    with open(r"D:\Data_sets\ShapeNetCore_v2\ShapeNet64_Chair\ShapeNet64_Chair_binvox_3_centered\model_normalized_25_clean.binvox",
            'rb') as fp:
        shape_in_2 = binvox_rw.read_as_3d_array(fp).data.reshape([1,64,64,64,1])

    shape_in=np.concatenate([shape_in_1, shape_in_2], axis=0)
    real_model_in = tf.placeholder(shape=[None, 64, 64, 64, 1], dtype=tf.float32)  # Real 3D voxel objects
    center=tf.constant([0,0,0], dtype=tf.float32)
    #Diagonal (1,1,1 vector)
    dir=tf.constant([ 0.57735027,  0.57735027,  0.57735027], dtype=tf.float32)
    # dir=tf.constant([1,0,0], dtype=tf.float32)
    # dir=tf.constant([0,0,1], dtype=tf.float32)
    sym, base, base_z0, z, src_coords, wa, Ia, idx_b=symmetrised_feature_arbitraty_axis(real_model_in, center, dir)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sym_vox, base_vox, base_z0_vox, z_vox, src_coords_vox, wa_v, Ia_v, idx_b_v=sess.run([sym, base, base_z0, z, src_coords, wa, Ia, idx_b],
                                                                feed_dict={real_model_in:shape_in_1})
        print("wa " + str(wa_v.shape))
        print("Ia " + str(Ia_v.shape))
        print("idx b " + len(idx_b_v))
        print(sym_vox.shape)
        print(base_vox.shape)
        print(base_z0_vox.shape)
        print(z_vox.shape)
        print(src_coords_vox.shape)
        sym_vox=sym_vox>0
        # utils.save_binvox((sym_vox[1].reshape([64,64,64])), "D:/debug.binvox")
        visualise_voxel(sym_vox, show_grid=True, show_origin=True)
