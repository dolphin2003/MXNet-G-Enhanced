import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, "../../python"))
sys.path.insert(0, os.path.join(curr_path, "../../example/image-classification"))
import mxnet as mx
import logging
import argparse
import time
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="command for benchmark kv-store")
    parser.add_argument('--network', type=str, default="resnet",
                        help='the neural network to test')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='the gpus to be used, e.g "0,1,2,3"')
    parser.add_argument('--depth', type=int, default=152,
                        help='the depth of network, only valid for resnet')
    parser.add_argument('--kv-store', type=str, default='device',
                        help='the kvstore type')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size. should not affect the results')
    parser.add_argument('--num-batches', type=int, default=5,
                        help='number of batches to run')
    parser.add_argument('--disp-batches', type=int, default=1,
                        help='show averaged results for every n batches')
    parser.add_argument('--test-results', type=int, default=1,
                        help='if or not evalute the results correctness')
    parser.add_argument('--data-shape', type=str, default='128,3,224,224',
                        help='input data shape')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of classes')
    parser.add_argument('--optimizer', type=str, de