import os
import mxnet as mx
import numpy as np
import pickle as pkl
from mxnet.test_utils import *


def check_with_uniform(uf, arg_shapes, dim=None, npuf=None, rmin=-10, type_list=[np.float32]):
    """check function consistency with uniform random numbers"""
    if isinstance(arg_shapes, int):
        assert dim
        shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
        arg_shapes = [shape] * arg_shapes
    for dtype in type_list:
        ndarray_arg = []
        numpy_arg = []
        for s in arg_shapes:
            npy = np.random.uniform(rmin, 10, s).astype(dtype)
            narr = mx.nd.array(npy, dtype=dtype)
            ndarray_arg.append(narr)
            numpy_arg.append(npy)
        out1 = uf(*ndarray_arg)
        if npuf is None:
            out2 = uf(*numpy_arg).astype(dtype)
        else:
            out2 = npuf(*numpy_arg).astype(dtype)

        assert out1.shape == out2.shape
        if isinstance(out1, mx.nd.NDArray):
            out1 = out1.asnumpy()
        if dtype == np.float16:
            assert reldiff(out1, out2) < 1e-3
        else:
            assert reldiff(out1, out2) < 1e-6


def random_ndarray(dim):
    shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
    data = mx.nd.array(np.random.uniform(-10, 10, shape))
    return data

def test_ndarray_elementwise():
    np.random.seed(0)
    nrepeat = 10
    maxdim = 4
    all_type = [np.float32, np.float64, np.float16, np.uint8, np.int32]
    real_type = [np.float32, np.float64, np.float16]
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            check_with_uniform(lambda x, y: x + y, 2, dim, type_list=all_type)
            check_with_uniform(lambda x, y: x - y, 2, dim, type_list=all_type)
            check_with_uniform(lambda x, y: x * y, 2, dim, type_list=all_type)
            check_with_uniform(lambda x, y: x / y, 2, dim, type_list=real_type)
            check_with_uniform(lambda x, y: x / y, 2, dim, rmin=1, type_list=all_type)
            check_with_uniform(mx.nd.sqrt, 2, dim, np.sqrt, rmin=0)
            check_with_uniform(mx.nd.square, 2, dim, np.square, rmin=0)
            check_with_uniform(lambda x: mx.nd.norm(x).asscalar(), 1, dim, np.linalg.norm)

def test_ndarray_negate():
    npy = np.random.uniform(-10, 10, (2,3,4))
    arr = mx.nd.array(npy)
    assert reldiff(npy, arr.asnumpy()) < 1e-6
    assert reldiff(-npy, (-arr).asnumpy()) < 1e-6

    # a final check to make sure the negation (-) is not implemented
    # as inplace operation, so the contents of arr does not change after
    # we compute (-arr)
    assert reldiff(npy, arr.asnumpy()) < 1e-6


def test_ndarray_choose():
    shape = (100, 20)
    npy = np.arange(np.prod(shape)).reshape(shape)
    arr = mx.nd.array(npy)
    nrepeat = 3
    for repeat in range(nrepeat):
        indices = np.random.randint(shape[1], size=shape[0])
        assert same(npy[np.arange(shape[0]), indices],
                    mx.nd.choose_element_0index(arr, mx.nd.array(indices)).asnumpy())


def test_ndarray_fill():
    shape = (100, 20)
    npy = np.arange(np.prod(shape)).reshape(shape)
    arr = mx.nd.array(npy)
    new_npy = npy.copy()
    nrepeat = 3
    for repeat in range(nrepeat):
        indices = np.random.randint(shape[1], size=shape[0])
        val = np.