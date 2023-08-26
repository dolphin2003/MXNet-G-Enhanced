import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math

RNNState = namedtuple("RNNState", ["h"])
RNNParam = namedtuple("RNNParam", ["i2h_weight", "i2h_bias",
                                   "h2h_weight", "h2h_bias"])
RNNModel = namedtuple("RNNModel", ["rnn_exec", "symbol",
                                   "init_states", "last_states",
                                   "seq_data", "seq_labels", "seq_outputs",
                                   "param_blocks"])

def rnn(num_hidden, in_data, prev_state, param, seqidx, layeridx, dropout=0., batch_norm=False):
    if dropout > 0. :
        in_data = mx.sym.Dropout(data=in_data, p=dropout)
    i2h = mx.sym.FullyConnected(data=in_data,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    hidden = i2h + h2h

    hidden = mx.sym.Activation(data=hidden, act_type="tanh")
    if batch_norm == True:
        hidden = mx.sym.BatchNorm(data=hidden)
    return RNNState(h=hidden)



def rnn_unroll(num_rnn_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, dropout=0., batch_norm=False):

    embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_rnn_layer):
        param_cells.append(RNNParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                    i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                    h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                    h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = RNNState(h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_rnn_layer)

    loss_all = []
    for seqidx in range(seq_len):
        # embeding layer
        data = mx.sym.Variable("data/%d" % seqidx)

        hidden = mx.s