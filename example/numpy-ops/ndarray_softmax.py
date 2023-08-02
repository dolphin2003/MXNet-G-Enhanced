# pylint: skip-file
from data import mnist_iterator
import mxnet as mx
import numpy as np
import logging


class NDArraySoftmax(mx.operator.NDArrayOp):
    def __init__(self):
        super(NDArraySoftmax, self).__init__(False)
        self.fwd_kernel = None
        self.bwd_kernel = None
    
    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        if self.fwd_kernel is None:
            self.fwd_kernel = mx.rtc('softmax', [('x', x)], [('y', y)], """
int i = threadIdx.x + blockIdx.x*blockDim.x;
float max_x = x[i*x_dims[1]];
for (int j = 1; j < x_dims[1]; ++j) {
    if (max_x < x[i*x_dims[1]+j]) {
        max_x = x[i*x_dims[1]+j];
    }
}
float sum = 0.0f;
for (int j = 0; j < x_dims[1]; ++j) {
    sum += expf(x[i*x_dims[1]+j]-max_x);
}
for (int j = 0; j < x_dims[1]; ++j) {
    y[i*x_dims[1]+j] = expf(x[i*x_dims[1]+j]-max_x)/sum;
}
""")
        self.fwd_kernel.push([x], [y], (1, 1, 1), (x.shape[0], 1, 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        y = out_data