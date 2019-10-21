import os
import json
import sys
import tensorflow as tf
import numpy as np

with open(sys.argv[1], 'r') as fh:
    cfg = json.load(fh)

OUTPUT_DIR = cfg['output_dir']
LOGDIR = os.path.join(OUTPUT_DIR, "log")

os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(cfg['gpu'])
from model_HoloGAN import HoloGAN
from tools.utils import pp, show_all_variables


flags = tf.app.flags
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108] or [128] for celebA and lsun, [400] for chairs. Cats and Cars are already cropped")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce 64 or 128")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, lsun, chairs, shoes, cars, cats]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_float("train_size", np.inf, "Number of images to train-Useful when only a subset of the dataset is needed to train the model")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("rotate_azimuth", False, "Sample images with varying azimuth")
flags.DEFINE_boolean("rotate_elevation", False, "Sample images with varying elevation")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height
  if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop)

    model.build(cfg['build_func'])

    show_all_variables()

    if FLAGS.train:
        train_func = eval("model." + (cfg['train_func']))
        train_func(FLAGS)
    else:
      if not model.load(LOGDIR)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      model.sample_HoloGAN(FLAGS)


if __name__ == '__main__':
  tf.app.run()
