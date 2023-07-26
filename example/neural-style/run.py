import find_mxnet
import mxnet as mx
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
import argparse
from collections import namedtuple
from skimage import io, transform
from skimage.restoration import denoise_tv_chambolle

parser = argparse.ArgumentParser(description='neural style')

parser.add_argument('--model', type=str, default='vgg19',
                    choices = ['vgg'],
                    help = 'the pretrained model to use')
parser.add_argument('--content-image', type=str, default='input/IMG_4343.jpg',
                    help='the content image')
parser.add_argument('--style-image', type=str, default='input/starry_night.jpg',
                    help='the style image')
parser.add_argument('--stop-eps', type=float, default=.005,
                    help='stop if the relative chanage is less than eps')
parser.add_argument('--content-weight', type=float, default=10,
                    help='the weight for the content image')
parser.add_argument('--style-weight', type=float, default=1,
                    help='the weight for the style image')
parser.add_argument('--tv-weight', type=float, default=1e-2,
                    help='the magtitute on TV loss')
parser.add_argument('--max-num-epochs', type=int, default=1000,
                    help='the maximal number of training epochs')
parser.add_argument('--max-long-edge', type=int, default=600,
                    help='resize the content image')
parser.add_argument('--lr', type=float, default=.001,
                    help='the initial learning rate')
parser.add_argument('--gpu', type=int, default=0,
                    help='which gpu card to use, -1 means using cpu')
parser.add_argument('--output', type=str, default='output/out.jpg',
                    help='the output image')
parser.add_argument('--save-epochs', type=int, default=50,
                    help='save the output every n epochs')
parser.add_argument('--remove-noise', type=float, default=.02,
                    help='the magtitute to remove noise')

args = parser.parse_args()

def PreprocessContentImage(path, long_edge):
    img = io.imread(path)
    logging.info("load the content image, size = %s", img.shape[:2])
    factor = float(long_edge) / max(img.shape[:2])
    new_size = (int(img.shape[0] * factor), int(img.shape[1] * factor))
    resized_img = transform.resize(img, new_size)
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    logging.info("resize the content image to %s", new_size)
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PreprocessStyleImage(path, shape):
    img = io.imread(path)
    resized_img = transform.resize(im