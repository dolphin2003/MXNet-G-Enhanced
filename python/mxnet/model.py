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
        for k, p in enumerate(zip(arg_list, grad_list)):
            # faked an index here, to make optimizer create diff
            # state for the same index but on diff devs, TODO(mli)
            # use a better solution latter
            w, g = p
            updater(index*num_device+k, g, w)

train_accuracy_filename = None
train_accuracy_file_op = None
train_accuracy_epoch_filename = None
train_accuracy_epoch_file_op = None
train_accuracy_10per_filename = None
train_accuracy_10per_file_op = None
train_accuracy_top5_filename=None
train_accuracy_top5_file_op=None
train_accuracy_top5_epoch_filename=None
train_accuracy_top5_epoch_file_op=None
train_accuracy_10per = 10
train_interval_batch = 50
train_accuracy = 0.0
train_accuracy_top5=0.0
train_interval_time = 60
stop_train_record = False
current_time = 0

val_accuracy_filename = None
val_accuracy_epoch_filename = None
val_accuracy_epoch_file_op = None
val_accuracy_10per_filename = None
val_accuracy_10per_file_op = None
val_accuracy_file_op = None
val_accuracy_top5_filename=None
val_accuracy_top5_file_op=None
val_accuracy_top5_epoch_filename=None
val_accuracy_top5_epoch_file_op=None
val_accuracy_10per = 10
val_interval_batch = 50
val_accuracy = 0.0
val_accuracy_top5=0.0

def record_accuracy():
    """to record train accuracy and val accuracy every train_accuracy_interval_time secs(author: yegeyan)"""
    global train_accuracy_file_op
    global train_accuracy_epoch_file_op
    global train_accuracy_10per_file_op
    global train_accuracy_top5_file_op
    global train_accuracy_top5_epoch_file_op
    global train_accuracy
    global train_accuracy_top5
    global stop_train_record
    global current_time

    global val_accuracy_file_op
    global val_accuracy_epoch_file_op
    global val_accuracy_10per_file_op
    global val_accuracy_top5_file_op
    global val_accuracy_top5_epoch_file_op
    global val_accuracy_record_time
    global val_accuracy
    global val_accuracy_top5

    if stop_train_record:
        train_accuracy_file_op.write(time.strftime('end time: %Y-%m-%d %H:%M:%S\n',time.localtime(time.time())))
        train_accuracy_file_op.close()
        train_accuracy_epoch_file_op.write(time.strftime('end time: %Y-%m-%d %H:%M:%S\n',time.localtime(time.time())))
        train_accuracy_epoch_file_op.close()
        train_accuracy_10per_file_op.write(time.strftime('end time: %Y-%m-%d %H:%M:%S\n',time.localtime(time.time())))
        train_accuracy_10per_file_op.close()
        train_accuracy_top5_file_op.write(time.strftime('end time: %Y-%m-%d %H:%M:%S\n', time.localtime(time.time())))
        train_accuracy_top5_file_op.close()
        train_accuracy_top5_epoch_file_op.write(time.strftime('end time: %Y-%m-%d %H:%M:%S\n', time.localtime(time.time())))
        train_accuracy_top5_epoch_file_op.close()

        val_accuracy_file_op.write(time.strftime('end time: %Y-%m-%d %H:%M:%S\n',time.localtime(time.time())))
        val_accuracy_file_op.close()
        val_accuracy_epoch_file_op.write(time.strftime('end time: %Y-%m-%d %H:%M:%S\n',time.localtime(time.time())))
        val_accuracy_epoch_file_op.close()
        val_accuracy_10per_file_op.write(time.strftime('end time: %Y-%m-%d %H:%M:%S\n',time.localtime(time.time())))
        val_accuracy_10per_file_op.close()
        val_accuracy_top5_file_op.write(time.strftime('end time: %Y-%m-%d %H:%M:%S\n', time.localtime(time.time())))
        val_accuracy_top5_file_op.close()
        val_accuracy_top5_epoch_file_op.write(time.strftime('end time: %Y-%m-%d %H:%M:%S\n', time.localtime(time.time())))
        val_accuracy_top5_epoch_file_op.close()
        return

    if not math.isnan(train_accuracy):
        train_accuracy_file_op.write("%f\n" % train_accuracy)
        train_accuracy_file_op.flush()
    if not math.isnan(val_accuracy):
        val_accuracy_file_op.write("%f\n" % val_accuracy)
        val_accuracy_file_op.flush()
    if not math.isnan(train_accuracy_top5):
        train_accuracy_top5_file_op.write("%f\n" % train_accuracy_top5)
        train_accuracy_top5_file_op.flush()
    if not math.isnan(val_accuracy_top5):
        val_accuracy_top5_file_op.write("%f\n" % val_accuracy_top5)
        val_accuracy_top5_file_op.flush()
    current_time += train_interval_time
    t = threading.Timer(train_interval_time, record_accuracy)
    t.start()

max_stale = 0
min_iters = '0'
miniters_filename = None
miniters_file_op = None

is_straggler = False 

def _train_multi_device(symbol, ctx, arg_names, param_names, aux_names,
                        arg_params, aux_params,
                        begin_epoch, end_epoch, epoch_size, optimizer,
                        kvstore, update_on_kvstore,
                        train_data, eval_data=None, eval_metric=None, val_eval_metric=None,
                        epoch_end_callback=None, batch_end_callback=None,
                        logger=None, work_load_list=None, monitor=None,
                        eval_batch_end_callback=None, sym_gen=None):
    """Internal training function on multiple devices.
    This function will also work for single device as well.
    Parameters
    ----------
    symbol : Symbol
        The network configuration
    ctx : list of Context
        The training devices.
    arg_names: list of str
        Name of all arguments of the network.
    param_names: list of str
        Name of all trainable parameters of the network.
    aux_names: list of str
        Name of all auxiliary states of the network.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    begin_epoch : int
        The begining training epoch.
    end_epoch : int
        The end training epoch.
    epoch_size : int, optional
        Number of batches in a epoch. In default, it is set to
        ceil(num_train_examples / batch_size)
    optimizer : Optimizer
        The optimization algorithm
    train_data : DataIter
        Training data iterator.
    eval_data : DataIter
        Validation data iterator.
    eval_metric : EvalMetric
        An evaluation function or a list of evaluation functions.
    epoch_end_callback : callable(epoch, symbol, arg_params, aux_states)
        A callback that is invoked at end of each epoch.
        This can be used to checkpoint model each epoch.
    batch_end_callback : callable(BatchEndParams)
        A callback that is invoked at end of each batch.
        This can be used to measure speed, get result from evaluation metric. etc.
    kvstore : KVStore
        The KVStore
    update_on_kvstore : bool
        whether or not perform weight updating on kvstore
    logger : logging logger
        When not specified, default logger will be used.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as ctx
    monitor : Monitor, optional
        Monitor installed to executor,
        for monitoring outputs, weights, and gradients for debugging.
    Notes
    -----
    - This function will inplace update the NDArrays in arg_params and aux_states.
    """
    global train_accuracy_epoch_filename
    global train_accuracy_epoch_file_op
    global train_accuracy_10per_filename
    global train_accuracy_10per_file_op
    global train_accuracy_top5_filename
    global train_accuracy_top5_file_op
    global train_accuracy_top5_epoch_filename
    global train_accuracy_top5_epoch_file_op
    global train_accuracy
    global train_accuracy_10per
    global train_accuracy_top5
    global stop_train_record
    global is_straggler
    global current_time

    global val_accuracy_epoch_filename
    global val_accuracy_epoch_file_op
    global val_accuracy_10per_filename
    global val_accuracy_10per_file_op
    global val_accuracy_top5_filename
    global val_accuracy_top5_file_op
    global val_accuracy_top5_epoch_filename
    global val_accuracy_top5_epoch_file_op
    global val_accuracy
    global val_accuracy_10per
    global val_accuracy_top5

    global max_stale
    global min_iters
    global miniters_filename
    global miniters_file_op

    if logger is None:
        logger = logging
    executor_manager = DataParallelExecutorManager(symbol=symbol,
                                                   sym_gen=sym_gen,
                                                   ctx=ctx,
                                                   train_data=train_data,
                                                   param_names=param_names,
                                                   arg_names=arg_names,
                                                   aux_names=aux_names,
                                                   work_load_list=work_load_list,
                                                   logger=logger)

    if monitor:
        executor_manager.install_monitor(monitor)

    executor_manager.set_params(arg_params, aux_params)

    if not update_on_kvstore:
        updater = get_updater(optimizer)

    if kvstore:
        _initialize_kvstore(kvstore=kvstore,
                            param_arrays=executor_manager.param_arrays,
                            arg_params=arg_params,
                            param_names=executor_manager.param_names,
                            update_on_kvstore=update_on_kvstore)

    if update_on_kvstore:
        kvstore.set_optimizer(optimizer)

    # Now start training
    train_data.reset()
    total_batch = 0
    for epoch in range(begin_epoch, end_epoch):
        # Training phase
        tic = time.time()
        eval_metric.reset()
        val_eval_metric.reset() #yegeyan 2017.1.4
        nbatch = 0
        # Iterate over training data.
        while True:
            do_reset = True
            for data_batch in train_data:
                if kvstore.type == "dist_ssync" and not is_straggler:
                    wait_time = 0.0
                    while (not min_iters.isdigit()) or (min_iters.isdigit() and total_batch - int(min_iters) >= max_stale):
                        time.sleep(0.01)
                        miniters_file_op = open(miniters_filename, "r")
                        min_iters = miniters_file_op.read()
                        miniters_file_op.close()
                        wait_time += 0.01
                        if (abs(wait_time - 10.0) <= 1):
                            break

                executor_manager.load_data_batch(data_batch)

                if monitor is not None:
                    monitor.tic()

                executor_manager.forward(is_train=True)
                executor_manager.backward()

                if update_on_kvstore:
                    _update_params_on_kvstore(executor_manager.param_arrays,
                                              executor_manager.grad_arrays,
                                              kvstore)
                else:
                    _update_params(executor_manager.param_arrays,
                                   executor_manager.grad_arrays,
                                   updater=updater,
                                   num_device=len(ctx),
                                   kvstore=kvstore)

                if monitor is not None:
                    monitor.toc_print()

                # evaluate at end, so we can lazy copy
                executor_manager.update_metric(eval_metric, data_batch.label)

                nbatch += 1
                total_batch += 1
                if nbatch % train_interval_batch == 0:
                    train_accuracy = eval_metric.get()[1][0]
                    train_accuracy_top5=eval_metric.get()[1][1]
                    if train_accuracy * 100 >= train_accuracy_10per:
                        train_accuracy_10per_file_op.write(time.strftime('%H:%M:%S',time.localtime(time.time())))
                        train_accuracy_10per_file_op.write("   %f\n" % train_accuracy)
                        train_accuracy_10per_file_op.flush()
                        train_accuracy_10per = train_accuracy * 100 / 10 * 10 + 10

                if total_batch % val_interval_batch == 0:
                    val_eval_metric.reset()
                    eval_data.reset()

                    for i, eval_batch in enumerate(eval_data):
                        executor_manager.load_data_batch(eval_batch)
                        executor_manager.forward(is_train=False)
                        executor_manager.update_metric(val_eval_metric, eval_batch.label)

                    name_value = val_eval_metric.get_name_value()

                    for name, value in name_value:
                        if name =='accuracy':
                            val_accuracy = value;
                            if value * 100 > val_accuracy_10per:
                                val_accuracy_10per_file_op.write(time.strftime('%H:%M:%S',time.localtime(time.time())))
                                val_accuracy_10per_file_op.write("   %f\n" % value)
                                val_accuracy_10per_file_op.flush()
                                val_accuracy_10per = value * 100 / 10 * 10 + 10
                        if name=='top_k_accuracy_5':
                            val_accuracy_top5=value
                   