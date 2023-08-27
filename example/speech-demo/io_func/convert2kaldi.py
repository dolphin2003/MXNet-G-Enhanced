# Copyright 2013    Yajie Miao    Carnegie Mellon University 

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import sys

from StringIO import StringIO
import json
import utils.utils as utils
from model_io import string_2_array

# Various functions to convert models into Kaldi formats
def _nnet2kaldi(nnet_spec, set_layer_num = -1, filein='nnet.in',
               fileout='nnet.out', activation='sigmoid', withfinal=True):
    _nnet2kaldi_main(nnet_spec, set_layer_num=set_layer_num, filein=filein,
                    fileout=fileout, activation=activation, withfinal=withfinal, maxout=False)

def _nnet2kaldi_maxout(nnet_spec, pool_size = 1, set_layer_num = -1, 
                      filein='nnet.in', fileo