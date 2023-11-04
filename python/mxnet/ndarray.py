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

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.
        """
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=self.dtype)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(array)))
        source_array = np.ascontiguousarray(source_array, dtype=self.dtype)
        if source_array.shape != self.shape:
            raise ValueError('array shape do not match the shape of NDArray')
        check_call(_LIB.MXNDArraySyncCopyFromCPU(
            self.handle,
            source_array.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(source_array.size)))

    def _slice(self, start, stop):
        """Return a sliced NDArray that shares memory with current one.

        Parameters
        ----------
        start : int
            Starting index of slice.
        stop : int
            Finishing index of slice.
        """
        handle = NDArrayHandle()
        start = mx_uint(start) if start else mx_uint(0)
        stop = mx_uint(stop) if stop else mx_uint(self.shape[0])
        check_call(_LIB.MXNDArraySlice(
            self.handle, start, stop, ctypes.byref(handle)))
        return NDArray(handle=handle, writable=self.writable)

    def _copy_slice_to(self, axis, start, stop, target):
        """Copy a slice along an axis.

        Parameters
        ----------
        axis : int
            The axis along which to do slicing.
        start : int
            The starting index of the slice.
        stop : int
            The finishing index of the slice.
        target : NDArray or Context
            If an NDArray, must be pre-allocated with compatible shape.
            If a Context, a new NDArray will be created.

        Returns
        -------
        The sliced copy of the NDArray.
        """
        if isinstance(target, Context):
            shape = list(self.shape)
            shape[axis] = stop - start
            target = NDArray(_new_alloc_handle(shape, target, True, self.dtype))

        assert isinstance(target, NDArray)
        return _internal._copy_slice_to(self, axis, start, stop, out=target)

    def _at(self, idx):
        """Return a sub NDArray that shares memory with current one.

        Parameters
        ----------
        idx : int
            index of sub array.
        """
        handle = NDArrayHandle()
        idx = mx_uint(idx)
        check_call(_LIB.MXNDArrayAt(
            self.handle, idx, ctypes.byref(handle)))
        return NDArray(handle=handle, writable=self.writable)

    def reshape(self, new_shape):
        """Return a reshaped NDArray that shares memory with current one.

        Parameters
        ----------
        new_shape : iterable of int
            new shape of NDArray
        """
        handle = NDArrayHandle()
        check_call(_LIB.MXNDArrayReshape(self.handle,
                                         len(new_shape),
                                         c_array(ctypes.c_int, new_shape),
                                         ctypes.byref(handle)))
        return NDArray(handle=handle, writable=self.writable)

    # pylint: disable= undefined-variable
    def broadcast_to(self, shape):
        """ Broadcasting the current NDArray into the given shape. The semantics is
        the same with `numpy`'s broadcasting

        Parameters
        ---------
        shape : the shape to broadcast
            the broadcast shape
        """
        cur_shape = self.shape
        err_str = 'operands could not be broadcast together with remapped shapes' \
                  '[original->remapped]: {} and requested shape {}'.format(cur_shape, shape)
        if len(shape) < len(cur_shape):
            raise ValueError(err_str)
        cur_shape = (1,) * (len(shape) - len(cur_shape)) + cur_shape
        cur_shape_arr = np.array(cur_shape)
        broadcasting_axes = np.nonzero(cur_shape_arr != np.array(shape))
        if (cur_shape_arr[broadcasting_axes] != 1).any():
            raise ValueError(err_str)
        if cur_shape != self.shape:
            return broadcast_to(self.reshape(cur_shape), shape=shape)
        else:
            return broadcast_to(self, shape=tuple(shape))
    # pylint: enable= undefined-variable

    def wait_to_read(self):
        """Block until all pending writes operations on current NDArray are finished.

        This function will return when all the pending writes to the current
        NDArray finishes. There can still be pending read going on when the
        function returns.
        """
        check_call(_LIB.MXNDArrayWaitToRead(self.handle))

    @property
    def shape(self):
        """Get shape of current NDArray.

        Returns
        -------
        a tuple representing shape of current ndarray
        """
        ndim = mx_uint()
        pdata = ctypes.POINTER(mx_uint)()
        check_call(_LIB.MXNDArrayGetShape(
            self.handle, ctypes.byref(ndim), ctypes.byref(pdata)))
        return tuple(pdata[:ndim.value])

    @property
    def size(self):
        """Get size of current NDArray.

        Returns
        -------
        an int representing size of current ndarray
        """
        return np.prod(self.shape)

    @property
    def context(self):
        """Get context of current NDArray.

        Returns
        -------
        context : mxnet.Context
            The context of current NDArray.
        """
        dev_typeid = ctypes.c_int()
        dev_id = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetContext(
            self.handle, ctypes.byref(dev_typeid), ctypes.byref(dev_id)))
        return Context(Context.devtype2str[dev_typeid.value], dev_id.value)

    @property
    def dtype(self):
        """Get data type of current NDArray.

        Returns
        -------
        an numpy.dtype object representing type of current ndarray
        """
        mx_dtype = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetDType(
            self.handle, ctypes.byref(mx_dtype)))
        return _DTYPE_MX_TO_NP[mx_dtype.value]

    @property
    # pylint: disable= invalid-name, undefined-variable
    def T(self):
        """Get transpose of current NDArray"""
        if len(self.shape) != 2:
            raise ValueError('Only 2D matrix is allowed to be transposed')
        return transpose(self)
    # pylint: enable= invalid-name, undefined-variable

    def asnumpy(self):
        """Return