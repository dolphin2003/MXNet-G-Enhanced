# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-locals, too-many-arguments, too-many-statements
"""Executor manager"""
from __future__ import absolute_import

import logging
import numpy as np

from .base import mx_real_t
from . import ndarray as nd
from .context import cpu


def _split_input_slice(batch_size, work_load_list):
    """Get input slice from the input shape.
    Parameters
    ----------
    batch_size : int
        The number of samples in a mini-batch.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as ctx
    Returns
    -------
    slices : list of slice
        The split slices to get a specific slice.
    Raises
    ------
    ValueError
        If there are two many splits such that some slice can be empty.
    """
    total_work_load = sum(work_load_list)
    batch_num_list = [round(work_load * batch_size / total_work_load)
                      for work_load in work_load_list]
    batch_num_sum = sum(batch_num_list)
    if batch_num_sum < batch_size:
        batch_num_list[-1] += batch_size - batch_num_sum
    slices = []
    end = 0
    for batch_num in batch_num_list:
        begin = int(min((end, batch_size)))
        end = int(min((begin + batch_num, batch_size)))
        if begin >= end:
            raise ValueError('Too many slices such that some splits are empty')
        slices.append(slice(begin, end))
    return slices

def _check_arguments(symbol):
    """Check the argument names of symbol.
    This function checks the duplication of arguments in Symbol.
    The check is done for feedforward net for now.
    Parameters
    ----------
    symbol : Symbol
        The network configuration
    """
    arg_set = set()
    arg_names = symbol.list_arguments()
    for name in arg_names:
        if name in arg_set:
            raise ValueError(('Find duplicated argument name \"%s\", ' +
                              'please make the weight name non-duplicated(using name arguments), ' +
                              'arguments are %s') % (name, str(arg_names)))
        arg_set.add(name)

    aux_set = set()
    aux_names = symbol.list_auxiliary_states()
    for name in aux_names:
        if name in aux_set:
            raise ValueError(
                ('Find duplicated auxiliary param name \"%s\", ' +
                 'please make the weight name non-duplicated(using name arguments), ' +
                 'arguments are %s, auxiliary params are %s'
                ) % (name, str(arg_names), str(aux_names)))
        aux_set.add(name)

def _load_general(data, targets):
    """Load a list of arrays into a list of arrays specified by slices"""
    for d_src, d_targets in zip(data, targets):
        if isinstance(d_targets, nd.NDArray):
            d_src.copyto(d_targets)
        else:
            for slice_idx, d_dst in d_targets:
                d_src[slice_idx].copyto(d_dst)

def _load_data(batch, targets):
    """Load data into sliced arrays"""
    _load_general(batch.data, targets)

def _load_label(batch, targets):
    """Load label into sliced arrays"""
    _load_general(batch.label, targets)

# pylint: disable=too-many-branches
def _bind_exec(sym, ctx, input_shapes, param_names, need_grad=False,
               base_exec=None, shared_data_arrays=None, input_types=None, logger=logging):
    """bind executor for bucketing, potentially sharing data with an existing executor."""
    arg_shape, _, aux_shape = sym.infer_shape(**input_shapes)
    assert(arg_shape is not None)
    if input_types is None:
        input_types = {k: mx_real_t for k in input_shapes.keys()}
    arg_types, _, aux_types = sym.infer_type(**input_types)
    assert(arg_types is not None)

    arg_arrays = []
    grad_arrays = {} if need_grad != False else None

    arg_names = sym.list_arguments()

    if need_grad is False:
        need_grad = set()
    elif need_grad is True:
        need_grad = set(arg_names) - set(input_shapes.keys())
    elif isinstance(need_grad, set):
        pass
    else:
        raise AssertionError("need_grad must be boolean or set.")
    grad_req = {name:('write' if name in need_grad else 'null') for name in arg_names}


    # create or borrow arguments and gradients
    for i, name in enumerate(arg_names):
        if not name in param_names:
            # data or label
            if shared_data_arrays is not None and \
                    name in shared_data_arrays:
                arg_arr = shared_data_arrays[name]

                if np.prod(arg_arr.shape) >= np.prod(arg_shape[i]):
                    # good, we can share this memory
                    assert(arg_types[i] == arg_arr.dtype)
                    arg_arr = arg_arr.reshape(arg_shape[i])
                else:
                    logger.warning(('bucketing: data "%s" has a shape %s' % (name, arg_shape[i])) +
                                   (', which is larger than already allocated ') +
                                   ('shape %s' % (arg_arr.shape,)) +
                                   ('. Need to re-allocate. Consider putting ') +
                                   ('default_bucket_key to be the bucket taking the largest ') +
                                   ('input for better memory sharing.'))
                    arg_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])

                    # replace existing shared array because the new one is bigger
                    shared_data_arrays[name] = arg_arr
            else:
                arg_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])
                if shared_data_arrays is not None:
                    shared_data_arrays[name] = arg_arr

            arg_arrays.append(arg_arr)
        else:
            # model parameter
            if base_exec is None:
                arg_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])
                if name in need_grad:
                    grad_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])
                    grad_arrays[name] = grad_arr
            else:
                arg_arr = base_exec.arg_dict[name]
                assert arg_arr.shape == arg_shape[i]
                assert arg_arr.dtype == arg_types[i]
                if name in need_grad:
                    grad_arrays[name] = base_exec.grad_dict[name]
            arg_arrays.append(arg_arr)

    # create or borrow aux variables
    if base_exec is None:
        aux_arrays = [nd.zeros(s, ctx, dtype=t) for s, t in zip(aux_shape, aux_types)]
    else:
        for i, a in enumerate(base_exec.aux_arrays):
            assert aux_shape[i] == a.shape
            assert aux_types[i] == a.dtype

        aux_arrays = [a for a in base_exec.aux_arrays]

    executor = sym.bind(ctx=ctx, args=arg_arrays, args_grad=grad_arrays,
                        aux_states=aux_arrays,
                        grad_req=grad_req, shared_exec=base_exec)
    return executor

class DataParallelExecutorGroup(object):
    """A group of executors living on different devices, for data parallelization.

    Parameters
    ----------
    sym: Symbol
        The network configuration.
    arg_names: list of str
        Equals `sym.list_arguments()`
    param_names: list of str
        List of names of all trainable parameters.
    ctx: list of Context
        List of devices for training (data parallelization)
    slices: list of int
        Describes how the data parallelization splits data into different devices.
    train_data: DataIter (or DataBatch)
        The dataset for training. It could be any object with `provide_data` and
        `provide_label` properties. Loading of actual data is not necessarily needed
        at this stage.
    shared_grop: DataParallelExecutorGroup
        An existing executor group, if to share parameters with it.
    """
    def __init__(self, sym, arg_names, param_names, ctx, slices, train_data, shared_group=None):
        # make sure the architecture is valid
        _check_arguments(sym)

        if shared_group is None:
            self.shared_data_arrays = [{} for _ in ctx]
        else:
            self.shared_data_arrays = shared_group.shared_data_arrays

        self.data_names = [x[0] for x in train_data.provide_data]
        self.label_names = [x[0] for x in train_data.provide_label]
        self.aux_names = sym.list_auxiliary_states()
        self.param_idx = [i for i in range(len(arg_names)) if arg_names[i] in param_names]
        self.param_names = [arg_names[i] for i in self.param_idx]

        self.train_execs = []
        for i, ctxi in enumerate(ctx):
            data_shapes = {k: tuple([slices[i].stop-slices[i].start] + list(v[1:]))
                           for k, v in train_data.provide_data + train_data.provide_label}
            shared_exec = None if shared_group is None else shared_group.train_execs[i]
            train_exec = _bind_exec(sym, ctxi, data_shapes, self.param_names,
                                    need_grad=True, base_exec=shared_exec,
                                    shared_data_arrays=self.shared_data_arrays[i])
            self.train_execs.append(train_exec)

        # data structure
        self.data_arrays = [[(slices[i], e.arg_dict[name]) for i, e in enumerate(self.train_execs)]
                            for name in self.data_names]
        self.label_arrays = [[(slices[i], e.arg_dict[name]) for i, e in enumerate(self.train_execs)]
                             for name in self.label_names]

        self.param_arrays = [[e.arg_arrays[i] for e in self.train_execs]
                             for i in self.param_idx]
        self.grad_arrays = [[e.grad_arrays[i] for e in self.train_execs]
                            for i in self.param_idx]

        self.aux_arrays = [[e.aux_arrays[i] for e in self.train_execs]
                           for i in range(len(self.aux_names))]

        self.slices = slices

    def load_data_batch(self, data_batch):
        """ load data and labels into arrays """
        _load_data(data_batch, self.data_arrays)
        _load_label(data_batch, self.label_arrays)

    def forward(self, is_train=False):
        """ Perform a forward pass on each executor """
        for texec in self.train_execs:
            texec.forward(is_train=is_train)

    def backward(self):
        """ Perform a backward pass on each executor """
        for texec in self.train_execs:
            texec.backward()

    def update_metric(self, metric, labels):
        """ Update evaluation metric with label and cu