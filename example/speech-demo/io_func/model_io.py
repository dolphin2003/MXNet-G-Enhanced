import numpy as np
import os
import sys
import logging

from StringIO import StringIO
import json


from datetime import datetime

from kaldi_parser import *
import utils.utils as utils

# nicer interface for file2nnet, nnet2file

def load(model, filename, gradients, num_hidden_layers=-1, with_final=True, factors=None):
    _file2nnet(model.sigmoid_layers, set_layer_num = num_hidden_layers,
        filename=filename, activation="sigmoid", withfinal=with_final, factor=1.0, gradients=gradients, factors=factors)

def save(model, filename):
    _nnet2file(model.sigmoid_layers, set_layer_num = -1, filename=filename,
        activation="sigmoid", start_layer = 0, withfinal=True)

# convert an array to a string
def array_2_string(array):
    return array.astype('float32')

# convert a string to an array
def string_2_array(string):
    if isinstance(string, str) or isinstance(string, unicode):
        str_in = StringIO(string)
        return np.loadtxt(str_in)
    else:
        return string

def _nnet2file(layers, set_layer_num = -1, filename='nnet.out', activation='sigmoid', start_layer = 0, withfinal=True, input_factor = 0.0, factor=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]):
    logge