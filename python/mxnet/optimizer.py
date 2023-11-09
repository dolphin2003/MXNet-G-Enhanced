
# pylint: disable=fixme, invalid-name, unused-argument, too-many-arguments, no-name-in-module
"""Common Optimization algorithms with regularizations."""
import math
import ctypes
from .base import _LIB, check_call
from .base import c_array, mx_uint, mx_float, c_str
from .base import OptimizerHandle, OptimizerCreator
from .ndarray import NDArray, zeros, clip, sqrt, square
from .random import normal
import time
import numpy as np

class Optimizer(object):
    """Base class of all optimizers."""
    opt_registry = {}

    @staticmethod
    def register(klass):
        """Register optimizers to the optimizer factory"""
        assert(isinstance(klass, type))
        name = klass.__name__.lower()
        if name in Optimizer.opt_registry:
            print('WARNING: New optimizer %s.%s is overriding '
                  'existing optimizer %s.%s' % (
                      klass.__module__, klass.__name__,
                      Optimizer.opt_registry[name].__module__,
                      Optimizer.opt_registry[name].__name__))
        Optimizer.opt_registry[name] = klass
        return klass

    @staticmethod
    def create_optimizer(name, rescale_grad=1, **kwargs):
        """Create an optimizer with specified name.

        Parameters
        ----------
        name: str
            Name of required optimizer. Should be the name
            of a subclass of Optimizer. Case insensitive.

        rescale_grad : float
            Rescaling factor on gradient.

        kwargs: dict
            Parameters for optimizer

        Returns
        -------
        opt : Optimizer
            The result optimizer.
        """
        if name.lower() in Optimizer.opt_registry:
            return Optimizer.opt_registry[name.lower()](
                rescale_grad=rescale_grad,
                **kwargs)
        else:
            raise ValueError('Cannot find optimizer %s' % name)

    @staticmethod
    def _init_cc_optimizer(name, param_keys, param_vals):
        """Initialize handle to C++ optimizer.

        Parameters
        ----------
        name : str
            name of the optimizer registered with MXNET_REGISTER_OPTIMIZER
        param_keys : list of str
            list of argument names passed to Init(kwargs)
        param_vals : list
            corresponding values

        Returns
        -------
        handle : OptimizerHandle
            handle to the optimizer
        """
        creator = OptimizerCreator()
        check_call(_LIB.MXOptimizerFindCreator(c_str(name),
                                               ctypes.byref(creator)))
        assert creator, "Cannot find c++ implementation of optimizer \
                        registered with name "+name
        param_keys = c_array(ctypes.c_char_p, [c_str(s) for s in param_keys])
        param_vals = c_array(ctypes.c_char_p, [c_str(str(s)) for s in param_vals])
        handle = OptimizerHandle()
        check_call(_LIB.MXOptimizerCreateOptimizer(
            creator,
            mx_uint(len(param_keys)),
            param_keys, param_vals,
            ctypes.byref(handle)))
        return handle

    def __init__(self, rescale_grad=1., param_idx2name=None, wd=0.,
                 clip_gradient=None, learning_rate=0.01,
                 lr_scheduler=None, sym=None, begin_num_update=0):
        self.rescale_grad = rescale_grad
        self.lr = learning_rate
        self.lr_scheduler = lr_scheduler
        if lr_scheduler is not None:
            self.lr_scheduler.base_lr = learning_rate

        self.wd = wd
        self.lr_mult = {}
        self.wd_mult = {}
        self.begin_num_update = begin_num_update
        self.num_update = begin_num_update
        self._index_update_count = {}
        self.clip_gradient = clip_gradient
