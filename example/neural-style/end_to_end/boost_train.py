import sys
sys.path.insert(0, "../../mxnet/python")

import mxnet as mx
import numpy as np

import basic
import data_processing
import gen_v3
import gen_v4

# params
vgg_params = mx.nd.load("./vgg19.params")
style_weight = 1.2
content_w