import copy
import os
import mxnet as mx
import numpy as np
from common import models
import pickle as pkl

def test_symbol_basic():
    mlist = []
    mlist.append(models.mlp2())
    for m in mlist:
        m.list_arguments()
        m.list_outputs()

def test_symbol_compose():
    data = mx.symbol.Variable('data')
    net1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
    net1 = mx.symbol.FullyConnected(data=net1, name='fc2', num_hidden=100)
    net1.list_arguments() == ['data',
                              'fc1_weight', 'fc1_bias',
                              'fc2_weight', 'fc2_bias']

    net2 = mx.symbol.FullyConnected(name='fc3', num_hidden=10)
    net2 = mx.symbol.Activation(data=net2, act_type='relu')
    net2 = mx.symbol.FullyConnected(data=net2, name='fc4', num_hidden=20)
    #print(net2.debug_str())

    composed = net2(fc3_data=net1, name='composed')
    #print(composed.debug_str())
    multi_out = mx.symbol.Group([composed, net1])
    assert len(multi_out.list_outputs()) == 2


def test_symbol_copy():
    data = mx.symbol.Variable('data')
    data_2 = copy.deepcopy(data)
    data_3 = copy.copy(data)
    assert data.tojson() == data_2.tojson()
    assert data.tojson() == data_3.tojson()


def test_symbol_internal():
    data = mx.symbol.Variable('data')
    oldfc = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10