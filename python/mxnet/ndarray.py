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
 