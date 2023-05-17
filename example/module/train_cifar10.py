import logging
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image-classification")))

import find_mxnet
import mxnet as mx
import argparse
import train_model

my_dir = os.path.dirname(__file__)
default_data_dir = os.path.abspath(os.path.join(my_dir, '..', 'image-classification', 'cifar10')) + '/'

parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
parser.add_argument('--network', type=str, default='inception-bn-28-small',
                    help = 'the cnn to use')
parser.add_argument('--data-dir', type=str, default=default_data_dir,
                    help='the input data directory')
parser.add_argument('--gpus', type=str,
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=60000,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=.05,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add