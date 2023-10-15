# coding: utf-8
# pylint: disable=invalid-name, no-member
""" ctypes library of mxnet and helper functions """
from __future__ import absolute_import

import sys
import ctypes
import atexit
import numpy as np
from . import libinfo

__all__ = ['MXNetError']
#----------------------------
# library loading
#----------------------------
if sys.version_info[0] == 3:
    string_types = str,
    numeric_types = (float, int, np.float32, np.int32)
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode('utf-8')
else:
    string_types = basestring,
    numeric_types = (float, int, long, np.float32, np.int32)
    py_str = lambda x: x


class MXNetError(Exception):
    """Error that will be throwed by all mxnet functions"""
    pass

def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path()
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    # DMatrix functions
    lib.MXGetLastError.restype = ctypes.c_char_p
    return lib

# version number
__version__ = libinfo.__version__
# library instance of mxnet
_LIB = _load_lib()

# type definitions
mx_uint = ctypes.c_uint
mx_float = ctypes.c_float
mx_float_p = ctypes.POINTER(mx_float)
mx_real_t = np.float32
NDArrayHandle = ctypes.c_void_p
FunctionHandle = ctypes.c_void_p
SymbolCreatorHandle = ctypes.c_void_