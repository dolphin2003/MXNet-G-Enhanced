
# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from lstm import LSTMState, LSTMParam, lstm, bi_lstm_inference_symbol

class BiLSTMInferenceModel(object):
    def __init__(self,
                 seq_len,
                 input_size,
                 num_hidden,
                 num_embed,
                 num_label,
                 arg_params,
                 ctx=mx.cpu(),
                 dropout=0.):
        self.sym = bi_lstm_inference_symbol(input_size, seq_len,
                                            num_hidden,
                                            num_embed,
                                            num_label,
                                            dropout)
        batch_size = 1
        init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(2)]
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(2)]
        
        data_shape = [("data", (batch_size, seq_len, ))]

        input_shapes = dict(init_c + init_h + data_shape)
        self.executor = self.sym.simple_bind(ctx=mx.cpu(), **input_shapes)

        for key in self.executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.executor.arg_dict[key])

        state_name = []
        for i in range(2):
            state_name.append("l%d_init_c" % i)
            state_name.append("l%d_init_h" % i)

        self.states_dict = dict(zip(state_name, self.executor.outputs[1:]))
        self.input_arr = mx.nd.zeros(data_shape[0][1])

    def forward(self, input_data, new_seq=False):
        if new_seq == True:
            for key in self.states_dict.keys():
                self.executor.arg_dict[key][:] = 0.
        input_data.copyto(self.executor.arg_dict["data"])
        self.executor.forward()
        for key in self.states_dict.keys():
            self.states_dict[key].copyto(self.executor.arg_dict[key])
        prob = self.executor.outputs[0].asnumpy()
        return prob

