import tensorflow as tf
import numpy as np
import math
import time

def tf_repeat(x, n_repeats):
    #Repeat X for n_repeats time along 0 axis
    #Return a 1D tensor of total number of elements
    rep = tf.ones(shape=[1, n_repeats], dtype = 'int32')
    x = tf.matmul(tf.reshape(x, (-1,1)), rep)
    return tf.reshape(x, [-1])

def tf_interpolate(voxel, x, y, z, out_size):
    """
    Trilinear interpolation for batch of voxels
    :param voxel: The whole voxel grid
    :param x,y,z: indices of voxel
    :param output_size: output size of voxel
    :return:
    """
    batch_size = tf.shape(voxel)[0]
    height = tf.shape(voxel)[1]
    width = tf.shape(voxel)[2]
    depth = tf.shape(voxel)[3]
    n_channels = tf.shape(voxel)[4]

    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    z = tf.cast(z, 'float32')

    out_height = out_size[1]
    out_width = out_size[2]
    out_depth = out_size[3]
    out_channel = out_size[4]

    zero = tf.zeros([], dtype='int32')
    max_y = tf.cast(height - 1, 'int32')
    max_x = tf.cast(width - 1, 'int32')
    max_z = tf.cast(depth - 1, 'int32')

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), 'int32')
    z1 = z0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)

    #A 1D tensor of base indicies describe First index for each shape/map in the whole batch
    #tf.range(batch_size) * width * height * depth : Element to repeat. Each selement in the list is incremented by width*height*depth amount
    # out_height * out_width * out_depth: n of repeat. Create chunks of out_height*out_width*out_depth length with the same value created by tf.rage(batch_size) *width*height*dept
    base = tf_repeat(tf.range(batch_size) * width * height * depth, out_height * out_width * out_depth)

    #Find the Z element of each index

    base_z0 = base + z0 * width * height
    base_z1 = base + z1 * width * height
    #Find the Y element based on Z
    base_z0_y0 = base_z0 + y0 * width
    base_z0_y1 = base_z0 + y1 * width
    base_z1_y0 = base_z1 + y0 * width
    base_z1_y1 = base_z1 + y1 * width

    # Find the X element based on Y, Z for Z=0
    idx_a = base_z0_y0 + x0
    idx_b = base_z0_y1 + x0
    idx_c = base_z0_y0 + x1
    idx_d = base_z0_y1 + x1
    # Find the X element based on Y,Z for Z =1
    idx_e = base_z1_y0 + x0
    idx_f = base_z1_y1 + x0
    idx_g = base_z1_y0 + x1
    idx_h = base_z1_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    voxel_flat = tf.reshape(voxel, [-1, n_channels])
    voxel_flat = tf.cast(voxel_flat, 'float32')
    Ia = tf.gather(voxel_flat, idx_a)
    Ib = tf.gather(voxel_flat, idx_b)
    Ic = tf.gather(voxel_flat, idx_c)
    Id = tf.gather(voxel_flat, idx_d)
    Ie = tf.gather(voxel_flat, idx_e)
    If = tf.gather(voxel_flat, idx_f)
    Ig = tf.gather(voxel_flat, idx_g)
    Ih = tf.gather(voxel_flat, idx_h)

    # and finally calculate interpolated values
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    z0_f = tf.cast(z0, 'float32')
    z1_f = tf.cast(z1, 'float32')

    #First slice XY along Z where z=0
    wa = tf.expand_dims(((x1_f - x) * (y1_f - y) * (z1_f-z)), 1)
    wb = tf.expand_dims(((x1_f - x) * (y - y0_f) * (z1_f-z)), 1)
    wc = tf.expand_dims(((x - x0_f) * (y1_f - y) * (z1_f-z)), 1)
    wd = tf.expand_dims(((x - x0_f) * (y - y0_f) * (z1_f-z)), 1)
    # First slice XY along Z where z=1
    we = tf.expand_dims(((x1_f - x) * (y1_f - y) * (z-z0_f)), 1)
    wf = tf.expand_dims(((x1_f - x) * (y - y0_f) * (z-z0_f)), 1)
    wg = tf.expand_dims(((x - x0_f) * (y1_f - y) * (z-z0_f)), 1)
    wh = tf.expand_dims(((x - x0_f) * (y - y0_f) * (z-z0_f)), 1)


    output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id,  we * Ie, wf * If, wg * Ig, wh * Ih])
    return output

def tf_voxel_meshgrid(height, width, depth, homogeneous = False):
    with tf.variable_scope('voxel_meshgrid'):
        #Because 'ij' ordering is used for meshgrid, z_t and x_t are swapped (Think about order in 'xy' VS 'ij'
        z_t, y_t, x_t = tf.meshgrid(tf.range(depth, dtype = tf.float32),
                                    tf.range(height, dtype = tf.float32),
                                    tf.range(width, dtype = tf.float32), indexing='ij')
        #Reshape into a big list of slices one after another along the X,Y,Z direction
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        z_t_flat = tf.reshape(z_t, (1, -1))

        #Vertical stack to create a (3,N) matrix for X,Y,Z coordinates
        grid = tf.concat([x_t_flat, y_t_flat, z_t_flat], axis=0)
        if homogeneous:
            ones = tf.ones_like(x_t_flat)
            grid = tf.concat([grid, ones], axis = 0)
        return grid

def tf_rotation_around_grid_centroid(view_params, shapenet_viewer = False):
    #This function returns a rotation matrix around a center with y-axis being the up vector, and a scale matrix.
    #It first rotates the matrix by the azimuth angle (theta) around y, then around X-axis by elevation angle (gamma)
    #return a rotation matrix in homogenous coordinate
    #The default Open GL camera is to looking towards the negative Z direction
    #This function is suitable when the silhoutte projection is done along the Z direction

    batch_size = tf.shape(view_params)[0]

    azimuth    = tf.reshape(view_params[:, 0], (batch_size, 1, 1))
    elevation  = tf.reshape(view_params[:, 1], (batch_size, 1, 1))

    # azimuth = azimuth
    if shapenet_viewer == False:
        azimuth = (azimuth - tf.constant(math.pi * 0.5))

    #========================================================
    #Because tensorflow does not allow tensor item replacement
    #A new matrix needs to be created from scratch by concatenating different vectors into rows and stacking them up
    #Batch Rotation Y matrixes
    ones = tf.ones_like(azimuth)
    zeros = tf.zeros_like(azimuth)
    batch_Rot_Y = tf.concat([
        tf.concat([tf.cos(azimuth),  zeros, -tf.sin(azimuth), zeros], axis=2),
        tf.concat([zeros, ones,  zeros,zeros], axis=2),
        tf.concat([tf.sin(azimuth),  zeros, tf.cos(azimuth), zeros], axis=2),
        tf.concat([zeros, zeros, zeros, ones], axis=2)], axis=1)

    #Batch Rotation Z matrixes
    batch_Rot_Z = tf.concat([
        tf.concat([tf.cos(elevation),  tf.sin(elevation),  zeros, zeros], axis=2),
        tf.concat([-tf.sin(elevation), tf.cos(elevation),  zeros, zeros], axis=2),
        tf.concat([zeros, zeros, ones,  zeros], axis=2),
        tf.concat([zeros, zeros, zeros, ones], axis=2)], axis=1)


    transformation_matrix = tf.matmul(batch_Rot_Z, batch_Rot_Y)
    if tf.shape(view_params)[1] == 2:
        return transformation_matrix
    else:
    #Batch Scale matrixes:
        scale = tf.reshape(view_params[:, 2], (batch_size, 1, 1))
        batch_Scale= tf.concat([
            tf.concat([scale,  zeros,  zeros, zeros], axis=2),
            tf.concat([zeros, scale,  zeros, zeros], axis=2),
            tf.concat([zeros, zeros,  scale,  zeros], axis=2),
            tf.concat([zeros, zeros,  zeros, ones], axis=2)], axis=1)
    return transformation_matrix, batch_Scale

def tf_rotation_resampling(voxel_array, transformation_matrix, params, Scale_matrix = None, size=64, new_size=128):
    """
    Batch transformation and resampling function
    :param voxel_array: batch of voxels. Shape = [batch_size, height, width, depth, features]
    :param transformation_matrix: Rotation matrix. Shape = [batch_size, height, width, depth, features]
    :param size: original size of the voxel array
    :param new_size: size of the resampled array
    :return: transformed voxel array
    """
    batch_size = tf.shape(voxel_array)[0]
    n_channels = voxel_array.get_shape()[4].value
    target = tf.zeros([ batch_size, new_size, new_size, new_size])
    #Aligning the centroid of the object (voxel grid) to origin for rotation,
    #then move the centroid back to the original position of the grid centroid
    T = tf.constant([[1,0,0, -size * 0.5],
                  [0,1,0, -size * 0.5],
                  [0,0,1, -size * 0.5],
                  [0,0,0,1]])
    T = tf.tile(tf.reshape(T, (1, 4, 4)), [batch_size, 1, 1])

    # However, since the rotated grid might be out of bound for the original grid size,
    # move the rotated grid to a new bigger grid
    T_new_inv = tf.constant([[1, 0, 0, new_size * 0.5],
                             [0, 1, 0, new_size * 0.5],
                             [0, 0, 1, new_size * 0.5],
                             [0, 0, 0, 1]])
    T_new_inv = tf.tile(tf.reshape(T_new_inv, (1, 4, 4)), [batch_size, 1, 1])


    # Add the actual shifting in x and y dimension accoding to input param
    x_shift = tf.reshape(params[:, 3], (batch_size, 1, 1))
    y_shift = tf.reshape(params[:, 4], (batch_size, 1, 1))
    z_shift = tf.reshape(params[:, 5], (batch_size, 1, 1))
    # ========================================================
    # Because tensorflow does not allow tensor item replacement
    # A new matrix needs to be created from scratch by concatenating different vectors into rows and stacking them up
    # Batch Rotation Y matrixes
    ones = tf.ones_like(x_shift)
    zeros = tf.zeros_like(x_shift)

    T_translate = tf.concat([
        tf.concat([ones, zeros, zeros, x_shift], axis=2),
        tf.concat([zeros, ones, zeros, y_shift], axis=2),
        tf.concat([zeros, zeros, ones, z_shift], axis=2),
        tf.concat([zeros, zeros, zeros, ones], axis=2)], axis=1)
    total_M = tf.matmul(tf.matmul(tf.matmul(tf.matmul(T_new_inv, T_translate), Scale_matrix), transformation_matrix), T)


    try:
        total_M = tf.matrix_inverse(total_M)

        total_M = total_M[:, 0:3, :] #Ignore the homogenous coordinate so the results are 3D vectors
        grid = tf_voxel_meshgrid(new_size, new_size, new_size, homogeneous=True)
        grid = tf.tile(tf.reshape(grid, (1, tf.to_int32(grid.get_shape()[0]) , tf.to_int32(grid.get_shape()[1]))), [batch_size, 1, 1])
        grid_transform = tf.matmul(total_M, grid)
        x_s_flat = tf.reshape(grid_transform[:, 0, :], [-1])
        y_s_flat = tf.reshape(grid_transform[:, 1, :], [-1])
        z_s_flat = tf.reshape(grid_transform[:, 2, :], [-1])
        input_transformed = tf_interpolate(voxel_array, x_s_flat, y_s_flat, z_s_flat,[batch_size, new_size, new_size, new_size, n_channels])
        target= tf.reshape(input_transformed, [batch_size, new_size, new_size, new_size, n_channels])

        return target, grid_transform
    except tf.InvalidArgumentError:
        return None

def tf_3D_transform(voxel_array, view_params, size=64, new_size=128, shapenet_viewer=False):
    """
    Wrapper function to do 3D transformation
    :param voxel_array: batch of voxels. Shape = [batch_size, height, width, depth, features]
    :param transformation_matrix: Rotation matrix. Shape = [batch_size, height, width, depth, features]
    :param size: original size of the voxel array
    :param new_size: size of the resampled array
    :return: transformed voxel array
    """
    M, S = tf_rotation_around_grid_centroid(view_params[:, :3], shapenet_viewer=shapenet_viewer)
    target, grids = tf_rotation_resampling(voxel_array, M, params=view_params, Scale_matrix=S, size = size, new_size=new_size)
    return target

def generate_random_rotation_translation(batch_size, elevation_low=10, elevation_high=170, azimuth_low=0, azimuth_high=359,
                                         transX_low=-3, transX_high=3,
                                         transY_low=-3, transY_high=3,
                                         transZ_low=-3, transZ_high=3,
                                         scale_low=1.0, scale_high=1.0,
                                         with_translation=False, with_scale=False):
    params = np.zeros((batch_size, 6))
    column = np.arange(0, batch_size)
    azimuth = np.random.randint(azimuth_low, azimuth_high, (batch_size)).astype(np.float) * math.pi / 180.0
    temp = np.random.randint(elevation_low, elevation_high, (batch_size))
    elevation = (90. - temp.astype(np.float)) * math.pi / 180.0
    params[column, 0] = azimuth
    params[column, 1] = elevation

    if with_translation:
        shift_x = transX_low + np.random.random(batch_size) * (transX_high - transX_low)
        shift_y = transY_low + np.random.random(batch_size) * (transY_high - transY_low)
        shift_z = transZ_low + np.random.random(batch_size) * (transZ_high - transZ_low)
        params[column, 3] = shift_x
        params[column, 4] = shift_y
        params[column, 5] = shift_z

    if with_scale:
        scale = float(np.random.uniform(scale_low, scale_high))
        params[column, 2] = scale
    else:
        params[column, 2] = 1.0

    return params

