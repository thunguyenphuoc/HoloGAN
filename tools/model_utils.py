import numpy as np
import tensorflow as tf

import math
import os
import glob
import scipy.io


#=======================================================================================================================
#Helper functions to load pretrained weights
#=======================================================================================================================

def get_weight(weight_name, weight_dict):
    if weight_dict is None:
        print("Can't find weight")
        return None
    else:
        return weight_dict.get(weight_name)  # returns None if name is not found in dictionary

def load_weights(weight_dir):
    weight_path_all = glob.glob(os.path.join(weight_dir, "*.txt.npz"))
    pretrained_weight_dict = {}
    print(len(weight_path_all))
    for path in weight_path_all:
        with np.load(path) as data:
            layer_name = os.path.basename(path).split('.')[0]
            print(layer_name)
            pretrained_weight_dict[layer_name] = data['arr_0']
            print(data['arr_0'].shape)
    return pretrained_weight_dict


def load_z_mapping_function(z, output_channel, weight, bias, scope, act=None):
    with tf.variable_scope(scope) as sc:
        w = tf.get_variable('w', initializer=weight, trainable=False)
        b = tf.get_variable('biases', initializer=bias, trainable=False)
        if act == "lrelu":
            print ("LRELU")
            out = lrelu(tf.matmul(z, w) + b)
        else:
          out = act(tf.matmul(z, w) + b)
        return out[:, :output_channel], out[:, output_channel:]

def load_weights(weight_dir):
    weight_path_all = glob.glob(os.path.join(weight_dir, "*.txt.npz"))
    pretrained_weight_dict = {}
    print(len(weight_path_all))
    for path in weight_path_all:
        with np.load(path) as data:
            layer_name = os.path.basename(path).split('.')[0]
            print(layer_name)
            pretrained_weight_dict[layer_name] = data['arr_0']
    return pretrained_weight_dict

#=======================================================================================================================
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


#=====================================================================================================================================================

if __name__ == "__main__":
    #Testing random crop for voxels and corresponding images
    # path = r"D:\Data_sets\BaselFaceModel\BaselFaceModel\Generated\ply0_3.binvox"
    # with open(path, "rb") as fp:
    #     binvox = binvox_rw.read_as_3d_array(fp).data.astype(np.float32).reshape([64, 64, 64])
    #     binvox[32:, :, :] = 0
    #     utils.save_binvox(binvox, r"D:\Data_sets\BaselFaceModel\BaselFaceModel\Generated\ply0_3_half.binvox")
    # # # binvox = np.concatenate([binvox, binvox, binvox, binvox, binvox, binvox], axis = 0)
    # #
    # #
    # images = scipy.misc.imread(r"D:\Data_sets\ShapeNetCore_v2\ShapeNet64_Chair\SS\ShapeNet64_Chair_images_Contours\model_normalized_0_clean_23_3.png").astype(float).reshape((1, 512, 512, 3))
    # # images = np.concatenate([images, images, images, images, images, images], axis = 0)
    # #
    # crop_img
    # #
    # for i in range (crop_img.shape[0]):
    #     # utils.save_binvox(crop_vox[i].reshape((16, 16, 64)), "D:/BINVOX_cropped_{0}.binvox".format(i))
    #     scipy.misc.imsave("D:/cropped_{0}.png".format(i), crop_img[i])
    #

    # binvox = np.zeros((64, 64, 64))
    # binvox[32:, :32, 32:] = 1
    # binvox[:50, :20, :20] = 1
    # utils.save_binvox(binvox > 0, "D:/BINVOX_cropped.binvox")
    # import glob
    # images = np.zeros([4, 512, 512, 3])
    # path = glob.glob(os.path.join(os.path.join("D:\Data_sets\ShapeNetCore_v2\ShapeNet64_Chair\debug_pixelCNN_Normals_512", "*.png")))
    # print(len(path))
    # for i in range(len(path)):
    #     images[i] = scipy.misc.imread(path[i]).astype(np.float32)
    # crop_imgs = np_center_image_crop(images, 128)
    # for i in range(len(path)):
    #     name = os.path.basename(path[i]).split(".png")[0]
    #     scipy.misc.imsave("D:/crop_center_{0}.png".format(name), crop_imgs[i])
    # dict = scipy.io.loadmat(r"D:\Data_sets\BaselFaceModel\BaselFaceModel\Generated\alpha0.mat")
    # param = dict['alpha']
    # print(dict.keys())
    # print(param.shape)
    #===================================================================================
    #Test Phong shading
    #===================================================================================
    #===================================================================================
    #Test Phong shading
    #===================================================================================
    image_in = tf.placeholder(shape=[None, 512, 512, 3], dtype=tf.float32, name = "real_image_in") #2D render of voxel objects
    tf_ambient_in = tf.constant(0., dtype = tf.float32)
    tf_k_diffuse = tf.constant(0.7, dtype=tf.float32)
    # light_pos = tf.constant(np.array([[3., 2.5, 3], [1, 0, 0], [0, 1, 0], [-1, 0, 0]]), dtype=tf.float32)

    light_col = tf.constant(np.array([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0, 0, 1]]), dtype=tf.float32)
    batch_light_azimuth = tf.constant(np.tile(np.array([230 * math.pi / 180.0]), (4, 1)), dtype=tf.float32)
    light_pos = tf_generate_light_pos(batch_light_azimuth, (90-80) * math.pi / 180)
    phong = tf_phong_composite(image_in, light_dir=light_pos, ambient_in=tf_ambient_in, k_diffuse=tf_k_diffuse, k_intensity=light_col, with_mask=True)
    # img_np = np.expand_dims(scipy.misc.imread(
    #     r"D:\Projects\symmetryvae\Results\180514_FACE_NORMAL_gridParam_gradParamZTex_rdnInitTex_debugging\3_100_p299.0_t_75.0_los_0.04667_li_293.0_normal.jpg"), axis=0)/255.
    img_np = np.expand_dims(scipy.misc.imread(
        r"D:\Projects\symmetryvae\Results\180515_new_FACE_NORMAL_gridParam_gradParamZTex_rdnInitTex_Compare_DG_ICN_v2_debugging\4_300_p315.0_t_111.0_los_0.05654_li_398.0_normal.jpg"), axis=0)/255.

    img_np = np.tile(img_np[:, :, :, :3], (4, 1, 1, 1))

    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            # image, mask_out = sess.run([phong, mask], feed_dict={image_in: img_np})
            image = sess.run(phong, feed_dict={image_in: img_np})
            print(np.amax(image))
            image = np.clip(255 * image, 0, 255).astype(np.uint8)
            # mask_out = np.clip(255 * mask_out, 0, 255).astype(np.uint8)
            scipy.misc.imsave(os.path.join("D:/phong_tf_GT_v3.png"), image[0])
            # scipy.misc.imsave(os.path.join("D:/View for Chuan/phong_mask_tf.png"), np.squeeze(mask_out[0]))

    # with np.load(r"D:\Projects\SymmetryVAE\Results\00_CSPC_60\180328_3DAE_v4\z_all.txt.npz") as data:
    #     z_all = data['arr_0']
    #
    # z_mean = np.mean(z_all, axis = 0)
    # print(z_mean)
    # print(z_mean.shape)
    # z_cov = np.cov(z_all.T)
    # print(z_cov.shape)
    #
    # print(z_all.shape)
    #
    # np.savez(r"D:\Projects\SymmetryVAE\Results\00_CSPC_60\180328_3DAE_v4\z_all_mean.txt", z_mean)
    # np.savez(r"D:\Projects\SymmetryVAE\Results\00_CSPC_60\180328_3DAE_v4\z_all_cov.txt", z_cov)