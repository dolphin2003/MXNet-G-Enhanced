import sys
sys.path.insert(0, "../../mxnet/python/")

import mxnet as mx
import numpy as np
import model_vgg19 as vgg

class PretrainedInit(mx.init.Initializer):
    def __init__(self, prefix, params, verbose=False):
        self.prefix_len = len(prefix) + 1
        self.verbose = verbose
        self.arg_params = {k : v for k, v in params.items() if k.startswith("arg:")}
        self.aux_params = {k : v for k, v in params.items() if k.startswith("aux:")}
        self.arg_names = set([k[4:] for k in self.arg_params.keys()])
        self.aux_names = set([k[4:] for k in self.aux_params.keys()])

    def __call__(self, name, arr):
        key = name[self.prefix_len:]
        if key in self.arg_names:
            if self.verbose:
                print("Init %s" % name)
            self.arg_params["arg:" + key].copyto(arr)
        elif key in self.aux_params:
            if self.verbose:
                print("Init %s" % name)
            self.aux_params