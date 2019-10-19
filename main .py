import os
import json
import numpy as np
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(cfg['gpu'])
import tensorflow as tf


code_path    = os.environ['TEXTURENET_CODE']
data_path    = os.environ['TEXTURENET_DATA']
results_path = os.environ['TEXTURENET_RESULTS']

with open(sys.argv[1], 'r') as fh:
    cfg = json.load(fh)
IMAGE_PATH       = os.path.join(data_path, cfg['image_path'])
IMAGE_NAME_PATH = os.path.join(data_path, cfg['image_name_path'])
SAMPLE_SAVE = os.path.join(results_path, cfg['sample_save'])
MODEL_SAVE  = os.path.join(SAMPLE_SAVE, cfg['trained_model_name'])
LOGDIR = os.path.join(SAMPLE_SAVE, "log")


from model_HoloGAN import HoloGAN
from tools.utils import show_all_variables


flags = tf.app.flags
flags.DEFINE_integer("epoch", 50, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_integer("input_height",130, "The size of image to use (will be center cropped). [108] for celebA and lsun, [400] for chairs")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, lsun, chairs, shoes, cars, cats]")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", LOGDIR, "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", SAMPLE_SAVE, "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("composite", True, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  print("FLAGs " + str(FLAGS.dataset))
  with tf.Session(config=run_config) as sess:
    model = HoloGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=LOGDIR,
        sample_dir=SAMPLE_SAVE)

    model.build(cfg['build_func'])

    show_all_variables()

    if FLAGS.train:
        train_func = eval("model." + (cfg['train_func']))
        train_func(FLAGS)
    else:
      if not model.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
  tf.app.run()
