# pylint: skip-file
import mxnet as mx
from mxnet import misc
import numpy as np
import model
import logging
from solver import Solver, Monitor
try:
   import cPickle as pickle
except:
   import pickle

class AutoEncoderModel(model.MXModel):
    def setup(self, dims, sparseness_penalty=None, pt_dropout=None, ft_dropout=None, input_act=None, internal_act='relu', output_act=None):
        self.N = len(dims) - 1
        self.dims = dims
        self.stacks = []
        self.pt_dropout = pt_dropout
        self.ft_dropout = ft_dropout
        self.input_act = input_act
        self.internal_act = internal_act
        self.output_act = output_act

        self.data = mx.symbol.Variable('data')
        for i in range(self.N):
            if i == 0:
                decoder_act = input_act
                idropout = None
            else:
                decoder_act = internal_act
                idropout = pt_dropout
            if i == self.N-1:
                encoder_act = output_act
                odropout = None
            else:
                encoder_act = internal_act
                odropout = pt_dropout
            istack, iargs, iargs_grad, iargs_mult, iauxs = self.make_stack(i, self.data, dims[i], dims[i+1],
                                                sparseness_penalty, idropout, odropout, encoder_act, decoder_act)
            self.stacks.append(istack)
            self.args.update(iargs)
            self.args_grad.update(iargs_grad)
            self.args_mult.update(iargs_mult)
            self.auxs.update(iauxs)
        self.encoder, self.internals = self.make_encoder(self.data, dims, sparseness_penalty, ft_dropout, internal_act, output_act)
        self.decoder = self.make_decoder(self.encoder, dims, sparseness_penalty, ft_dropout, internal_act, input_act)
        if input_act == 'softmax':
            self.loss = self.decoder
        else:
            self.loss = mx.symbol.LinearRegressionOutput(data=self.decoder, label=self.data)

    def make_stack(self, istack, data, num_input, num_hidden, sparseness_penalty=None, idropout