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

    if nee