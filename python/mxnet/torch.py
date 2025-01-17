
# coding: utf-8
"""Interface for NDArray functions executed by torch backend.
Install torch and Compile with USE_TORCH=1 to use this module"""
from __future__ import absolute_import

import ctypes
import sys
from .base import _LIB
from .base import c_array, py_str, ctypes2docstring
from .base import mx_uint, mx_float, NDArrayHandle, FunctionHandle
from .base import check_call
from .ndarray import NDArray, _new_empty_handle

try:
    _LUAJIT = ctypes.CDLL("libluajit.so", mode=ctypes.RTLD_GLOBAL)
except OSError:
    _LUAJIT = None

# pylint: disable=too-many-locals, invalid-name
def _make_torch_function(handle):
    """Create a Torch function from the FunctionHandle."""
    # Get the property of function
    n_used_vars = mx_uint()
    n_scalars = mx_uint()
    n_mutate_vars = mx_uint()
    type_mask = ctypes.c_int()
    check_call(_LIB.MXFuncDescribe(
        handle,
        ctypes.byref(n_used_vars),
        ctypes.byref(n_scalars),
        ctypes.byref(n_mutate_vars),
        ctypes.byref(type_mask)))
    n_mutate_vars = n_mutate_vars.value
    n_used_vars = n_used_vars.value
    n_scalars = n_scalars.value
    type_mask = type_mask.value

    # Get the information from the function
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    ret_type = ctypes.c_char_p()

    check_call(_LIB.MXFuncGetInfo(
        handle, ctypes.byref(name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs),
        ctypes.byref(ret_type)))
    func_name = py_str(name.value)
    if not func_name.startswith('_th_'):
        return None
    param_str = ctypes2docstring(num_args, arg_names, arg_types, arg_descs)
    if n_mutate_vars > 1:
        res = ','.join(['res%d '%i for i in range(n_mutate_vars)])
    else:
        res = 'res '
    doc_str = (('Interface for Torch function {name}.\n' +
                'Invoke with\n{res}= mxnet.th.{name}(Parameters)\nor\n'+
                'mxnet.th.{name}({res}, Parameters).\n\n' +
                '{param_str}\n' +
                'Reference: ' +
                'https://github.com/torch/torch7/blob/master/doc/maths.md\n').format(
                    name=func_name[4:], param_str=param_str,
                    res=res))

    def generic_torch_function(*args, **kwargs):
        """Invoke this function by passing in parameters

        Parameters
        ----------
        *args
            Positional arguments of input scalars and NDArray

        Returns
        -------
        out : NDArray
            The result NDArray(tuple) of result of computation.
        """
        ndargs = []
        arg_format = ''
        value = ''
        for arg in args:
            if isinstance(arg, NDArray):
                ndargs.append(arg)
                arg_format += 'n'
                value += ','
            elif isinstance(arg, int):
                arg_format += 'i'
                value += str(arg) + ','
            elif isinstance(arg, str):
                arg_format += 's'
                value += str(arg) + ','
            elif isinstance(arg, float):
                arg_format += 'f'
                value += str(arg) + ','
            elif isinstance(arg, bool):
                arg_format += 'b'
                value += str(arg) + ','
        value = value[:-1]
        if len(ndargs) == n_used_vars:
            ndargs = [NDArray(_new_empty_handle()) for _ in range(n_mutate_vars)] + ndargs
            arg_format = 'n'*n_mutate_vars + arg_format
            value = ','*n_mutate_vars + value
        elif len(ndargs) == n_mutate_vars + n_used_vars:
            pass
        else:
            raise AssertionError(('Incorrect number of input NDArrays. ' +
                                  'Need to be either %d (inputs) or %d ' +
                                  '(output buffer) + %d (input)') %
                                 (n_used_vars, n_mutate_vars, n_used_vars))

        kwargs['format'] = arg_format
        kwargs['args'] = value

        for k in kwargs:
            kwargs[k] = str(kwargs[k])

        check_call(_LIB.MXFuncInvokeEx( \
                   handle, \
                   c_array(NDArrayHandle, [x.handle for x in ndargs[n_mutate_vars:]]), \
                   c_array(mx_float, []), \
                   c_array(NDArrayHandle, [x.handle for x in ndargs[:n_mutate_vars]]),
                   ctypes.c_int(len(kwargs)),
                   c_array(ctypes.c_char_p, kwargs.keys()),
                   c_array(ctypes.c_char_p, kwargs.values()),))
        if n_mutate_vars == 1:
            return ndargs[0]
        else:
            return ndargs[:n_mutate_vars]
    # End of function declaration
    ret_function = generic_torch_function
    ret_function.__name__ = func_name[4:]
    ret_function.__doc__ = doc_str
    return ret_function

# pylint: enable=too-many-locals, invalid-name

def _init_torch_module():
    """List and add all the torch backed ndarray functions to current module."""
    plist = ctypes.POINTER(FunctionHandle)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListFunctions(ctypes.byref(size),
                                    ctypes.byref(plist)))

    module_obj = sys.modules[__name__]
    for i in range(size.value):
        hdl = FunctionHandle(plist[i])
        function = _make_torch_function(hdl)
        # if function name starts with underscore, register as static method of NDArray
        if function is not None:
            setattr(module_obj, function.__name__, function)

# Initialize the NDArray module
_init_torch_module()