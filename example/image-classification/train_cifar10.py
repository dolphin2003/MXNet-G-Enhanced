import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_model
import socket #2016.10.6
import linecache

parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
parser.add_argument('--network', type=str, default='inception-bn-28-small',
    