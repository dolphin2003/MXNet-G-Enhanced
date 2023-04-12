# How to Create New Operations (Layers)

This note will walk you through the process of creating new MXNet operations (or layers).

We try to do our best to provide high speed operators for most common use cases. However, if you do find yourself in need of custom layers, like a novel loss for your research, you have two options:

* ~~(Deprecated) Use native language and it's matrix library (e.g. numpy in Python). This requires least effort and knowledge of MXNet. But impairs performance as it is CPU based.~~

* ~~(Deprecated) Use native language, mxnet.rtc and mxnet.ndarray. This gives you most of the performance of 3) and most of the convenience of 1), but requires more knowledge of MXNet. You can write CUDA kernels in python and compile with during runtime.~~

* 1) Use CustomOp to write new operators in frontend language (i.e. Python) that runs on cpu or gpu. Depending on your implementation, this can range from very fast to very slow.

* 2) Use C++/MShadow(CUDA). This can be difficult if you are not familiar with MXNet, mashadow or Cuda, but it will give you the best performance.

## CustomOp
Implementing an operator in Python is similar to creating one in C++ but simpler. Let's create a softmax operator for example. We start by subclassing `mxnet.operator.CustomOp` and then override a few methods:
```python
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np

class Softmax(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], mx.nd.array(y))
```
Here we defined the computation for forward pass of our operator. The forward function takes a list of input and a list of output NDArrays. Here we called .asnumpy() on the input NDArray to convert it to cpu based numpy arrays for convenience.

Keep in mind that this can be very slow. If you want the best performance, keep data in NDArray format and use operations under mx.nd to do the computation.

At the end, we used CustomOp.assign to assign the resulting array y to out_data[0]. It handles assignment based on the value of req, which can be 'write', 'add' or 'null'.

Then we do the same for backward:
```python
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[1].asnumpy().ravel().astype(np.int)
        y = out_data[0].asnumpy()
        y[np.arange(l.shape[0]), l] -= 1.0
        self.assign(in_grad[0