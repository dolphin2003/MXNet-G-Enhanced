# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, too-many-lines
# pylint: disable=too-many-branches, too-many-statements
"""MXNet model module"""
from __future__ import absolute_import

import time
import logging
from collections import namedtuple
import numpy as np

from . import io
from . import nd
from . import symbol as sym
from . import optimizer as opt
from . import metric
from . import kvstore as kvs
from .context import Context, cpu
from .initializer import Uniform
from .optimizer import get_updater
from .executor_manager import DataParallelExecutorManager, _check_arguments, _load_data
import threading 
import math 
import getpass
import socket
import struct
import fcntl
BASE_ESTIMATOR = object

try:
    from sklearn.base import BaseEstimator
    BASE_ESTIMATOR = BaseEstimator
except ImportError:
    SKLEARN_INSTALLED = False

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'eval_metric',
                            'locals'])

def _create_kvstore(kvstore, num_device, arg_params):
    """Create kvstore
    This function select and create a proper kvstore if given the kvstore type

    Parameters
    ----------
    kvstore : KVStore or str
        The kvstore
    num_device : int
        The number of devices
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    """
    update_on_kvstore = True
    if kvstore is None:
        kv = None
    elif isinstance(kvstore, kvs.KVStore):
        kv = kvstore
    elif isinstance(kvstore, str):
        # create kvstore using the string type
        if num_device is 1 and 'dist' not in kvstore:
            # no need to use kv for single device and single machine
            kv = None
        else:
            kv = kvs.create(kvstore)
            if kvstore is 'local':
            # automatically select a proper local
                max_size = max(np.prod(param.shape) for param in
                               arg_params.values())
                if max_size > 1024 * 1024 * 16:
                    update_on_kvstore = False
    else:
        raise TypeError('kvstore must be KVStore, str or None')

    if kv is None:
        update_on_kvstore = False

    return (kv, update_on_kvstore)

def _initialize_kvstore(kvstore, param_arrays, arg_params, param_names,
                        update_on_kvstore):
    """ Initialize kvstore"""
    for idx, param_on_devs in enumerate(param_arrays):
        kvstore.init(idx, arg_params[param_names[idx]])

        if update_on_kvstore:
            kvstore.pull(idx, param_on_devs, priority=-idx)

def _update_params_on_kvstore(param_arrays, grad_arrays, kvstore):
    """ Perform update of param_arrays from grad_arrays on kvstore."""
    #key_num = 0;
    for index, pair in enumerate(zip(param_arrays, grad_arrays)):
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        # push gradient, priority is negative index
        kvstore.push(index, grad_list, priority=-index)
        #key_num += 1
        # pull back the weights
        kvstore.pull(index, arg_list, priority=-index)

def _update_params(param_arrays, grad_arrays, updater, num_device,
                   kvstore=None):
    """ Perform update of param_arrays from grad_arrays not on kvstore."""
    for index, pair in enumerate(zip(param_arrays, grad_arrays)):
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        if kvstore:
            # push gradient, priority is negative index
            kvstore.push(index, grad_list, priority=-index)
            # pull back the sum gradients, to the same locations.
            kvstore.pull(index, grad_list, priority=-index)
        for