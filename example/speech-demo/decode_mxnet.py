import re
import sys
sys.path.insert(0, "../../python")
import time
import logging
import os.path

import mxnet as mx
import numpy as np

from lstm_proj import lstm_unroll
from io_util import BucketSentenceIter, TruncatedSentenceIter, SimpleIter, DataReadStream
from config_util import parse_args, get_checkpoint_path, parse_contexts

from io_func.feat_readers.writer_kaldi import KaldiWriteOut

# some constants
METHOD_BUCKETING = 'bucketing'
METHOD_TBPTT = 'truncated-bptt'
METHOD_SIMPLE = 'simple'

def prepare_data(args):
    batch_size = args.config.getint('train', 'batch_size')
    num_hidden = args.config.getint('arch', 'num_hidden')
    num_hidden_proj = args.config.getint('arch', 'num_hidden_proj')
    num_lstm_layer = args.config.getint('arch', 'num_lstm_layer')

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    if num_hidden_proj > 0:
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden_proj)) for l in range(num_lstm_layer)]
    else:
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]

    init_states = init_c + init_h

    file_test = args.config.get('data', 'test')
    file_label_mean =  args.config.get('data', 'label_mean')
    file_format = args.config.get('data', 'format')
    feat_dim = args.config.getint('data', 'xdim')
    label_dim = args.config.getint('data', 'ydim')

    test_data_args = {
            "gpu_chunk": 32768,
            "lst_file": file_test,
            "file_format": file_format,
            "separate_lines":True,
            "has_labels":False
            }

    label_mean_args = {
            "gpu_chunk": 32768,
            "lst_file": file_label_mean,
            "file_format": file_format,
            "separate_lines":True,
            "has_labels":False
            }

    test_sets = DataReadStream(test_data_args, feat_dim)
    label_mean_sets = DataReadStream(label_mean_args, lab