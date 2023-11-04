# coding: utf-8
# pylint: disable= too-many-lines, redefined-builtin, protected-access
"""NDArray API of mxnet."""
from __future__ import absolute_import
from __future__ import division

import ctypes
import warnings
import sys
import functools
import operator
import numpy as np
from .base import _LIB, string_types, numeric_types
from .base import c_array, mx_float, py_str, c_str, mx_real_t
from .base import mx_uint, NDArrayHandle, FunctionHandle
from .base import ctypes2buffer
from .base import check_call, ctypes2docstring
from .context import Context
from . import _ndarray_internal as _internal

# pylint: disable= no-member
_DTYPE_NP_TO_MX = {
    np.float32 : 0,
    np.float64 : 1,
    np.float16 : 2,
    np.uint8   : 3,
    np.int32   : 4
}

_DTYPE_MX_TO_NP = {
    0 : np.float32,
    1 : np.float64,
    2 : np.float16,
    3 : np.uint8,
    4 : np.int32
}
# pylint: enable= no-member

def _new_empty_handle():
    """Return a new empty handle.

    Empty handle can be used to hold result

    Returns
    -------
    a new empty ndarray handle
    """
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateNone(ctypes.byref(hdl)))
    return hdl

def _new_alloc_handle(shape, ctx, delay_alloc, dtype=mx_real_t):
    """Return a new handle with specified shape and context.

    Empty handle is only used to hold results

    Returns
    -------
    a new empty ndarray handle
    """
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateEx(
        c_array(mx_uint, shape),
        mx_uint(len(shape)),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(delay_alloc)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])),
        ctypes.byref(hdl)))
    return hdl

def waitall():
    """Wait all async operation to finish in MXNet

    This function is used for benchmark only
    """
    check_call(_LIB.MXNDArrayWaitAll())

class NDArray(object):
    """NDArray object in mxnet.

    NDArray is basic ndarray/Tensor like data structure in mxnet.
    """
    # pylint: disable= no-member
    def __init__(self, handle, writable=True):
        """initialize a new NDArray

        Parameters
        ----------
        handle : NDArrayHandle
            NDArray handle of C API
        """
        assert isinstance(handle, NDArrayHandle)
        self.handle = handle
        self.writable = writable

    def __del__(self):
        check_call(_LIB.MXNDArrayFree(self.handle))

    def __add__(self, other):
        return add(self, other)

    def __iadd__(self, other):
        if not self.writable:
            raise ValueError('trying to add to a readonly NDArray')
        if isinstance(other, NDArray):
            return _internal._plus(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._plus_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return subtract(self, other)

    def __isub__(self, other):
        if not self.writable:
            raise ValueError('trying to subtract from a readonly NDArray')
        if isinstance(other, NDArray):
            return _internal._minus(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._minus_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __neg__(self):
        return _internal._mul_scalar(self, -1.0)

    def __imul__(self, other):
        if not self.writable:
            raise ValueError('trying to multiply to a readonly NDArray')
        if isinstance(other, NDArray):
            return _internal._mul(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._mul_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return divide(self, other)

    def __rdiv__(self, other):
        return divide(other, self)

    def __idiv__(self, other):
        if not self.writable:
            raise ValueError('trying to divide from a readonly NDArray')
        if isinstance(other, NDArray):
            return _internal._div(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _internal._div_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __pow__(self, other):
        return power(self, other)

    def __rpow__(self, other):
        return power(other, self)

    def __getstate__(self):
        this = self.__dict__.copy()
        handle = this['handle']
        if handle is not None:
            length = ctypes.c_size_t()
            cptr = ctypes.POINTER(ctypes.c_char)()
            check_call(_LIB.MXNDArraySaveRawBytes(self.handle,
                                                  ctypes.byref(length),
                                                  ctypes.byref(cptr)))
            this['handle'] = ctypes2buffer(cptr, length.value)
        return this

    def __setstate__(self, state):
        handle = state['handle']
        if handle is not None:
            buf = handle
            handle = NDArrayHandle()
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            length = ctypes.c_size_t(len(buf))
            check_call(_LIB.MXNDArrayLoadFromRawBytes(ptr, length, ctypes.byref(handle)))
            state['handle'] = handle
        self.__dict__.update(state)

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if not self.writable:
            raise ValueError('trying to assign to a readonly NDArray')
        if isinstance(in_slice, int):
            sliced_arr = self._at(in_slice)
            sliced_arr[:] = value
            return
        if not isinstance(in_slice, slice) or in_slice.step is not None:
            raise ValueError('NDArray only support continuous slicing on axis 0')
        if in_slice.start is not None or in_slice.stop is not None:
            sliced_arr = self._slice(in_slice.start, in_slice.stop)
            sliced_arr[:] = value
            return
        if isinstance(value, NDArray):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, numeric_types):
            _internal._set_value(float(value), out=self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self._sync_copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def __getitem__(self, in_slice):
        """Get ndarray"""
        if isinstance(in_slice, int):
            return self._at(in_slice)
        if not isinstance(in_slice, slice) or in_slice.step is not None:
            raise ValueError('NDArray only support continuous slicing on axis 0')
        if in_slice.start is not None or in_slice.stop is not None:
            return self._slice(in_slice.start, in_slice.stop)
        else:
            return self

    def _sync_copyfrom(self, source_array):
        """Peform an synchronize copy from the array.

     