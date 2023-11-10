
# coding: utf-8
"""Tools for testing."""
# pylint: disable=invalid-name, no-member, too-many-arguments, too-many-locals, too-many-branches, too-many-statements, broad-except, line-too-long
from __future__ import absolute_import, print_function, division
import time
import numpy as np
import numpy.testing as npt
import mxnet as mx
_rng = np.random.RandomState(1234)


def np_reduce(dat, axis, keepdims, numpy_reduce_func):
    """Compatible reduce for old version numpy

    Parameters
    ----------
    dat : np.ndarray
        Same as Numpy

    axis : None or int or list-like
        Same as Numpy

    keepdims : bool
        Same as Numpy

    numpy_reduce_func : function
        Numpy reducing function like `np.sum` or `np.max`
    """
    if isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis) if axis is not None else range(len(dat.shape))
    ret = dat
    for i in reversed(sorted(axis)):
        ret = numpy_reduce_func(ret, axis=i)
    if keepdims:
        keepdims_shape = list(dat.shape)
        for i in axis:
            keepdims_shape[i] = 1
        ret = ret.reshape(tuple(keepdims_shape))
    return ret


def same(a, b):
    """Test if two numpy arrays are the same

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    """
    return np.array_equal(a, b)


def reldiff(a, b):
    """Calculate the relative difference between two input arrays

    Calculated by :math:`\\frac{|a-b|^2}{|a|^2 + |b|^2}`

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    """
    diff = np.sum(np.abs(a - b))
    norm = np.sum(np.abs(a)) + np.sum(np.abs(b))
    if diff == 0:
        return 0
    ret = diff / norm
    return ret


def _parse_location(sym, location, ctx):
    """Parse the given location to a dictionary

    Parameters
    ----------
    sym : Symbol
    location : None or list of np.ndarray or dict of str to np.ndarray

    Returns
    -------
    dict of str to np.ndarray
    """
    assert isinstance(location, (dict, list, tuple))
    if isinstance(location, dict):
        if set(location.keys()) != set(sym.list_arguments()):
            raise ValueError("Symbol arguments and keys of the given location do not match."
                             "symbol args:%s, location.keys():%s"
                             % (str(set(sym.list_arguments())), str(set(location.keys()))))
    else:
        location = {k: v for k, v in zip(sym.list_arguments(), location)}
    location = {k: mx.nd.array(v, ctx=ctx) for k, v in location.items()}
    return location


def _parse_aux_states(sym, aux_states, ctx):
    """

    Parameters
    ----------
    sym : Symbol
    aux_states : None or list of np.ndarray or dict of str to np.ndarray

    Returns
    -------
    dict of str to np.ndarray
    """
    if aux_states is not None:
        if isinstance(aux_states, dict):
            if set(aux_states.keys()) != set(sym.list_auxiliary_states()):
                raise ValueError("Symbol aux_states names and given aux_states do not match."
                                 "symbol aux_names:%s, aux_states.keys:%s"
                                 % (str(set(sym.list_auxiliary_states())),
                                    str(set(aux_states.keys()))))
        elif isinstance(aux_states, (list, tuple)):
            aux_names = sym.list_auxiliary_states()
            aux_states = {k:v for k, v in zip(aux_names, aux_states)}
        aux_states = {k: mx.nd.array(v, ctx=ctx) for k, v in aux_states.items()}
    return aux_states


def numeric_grad(executor, location, aux_states=None, eps=1e-4, use_forward_train=True):
    """Calculates a numeric gradient via finite difference method.

    Class based on Theano's `theano.gradient.numeric_grad` [1]

    Parameters
    ----------
    executor : Executor
        exectutor that computes the forward pass
    location : list of numpy.ndarray or dict of str to numpy.ndarray
        Argument values used as location to compute gradient
        Maps the name of arguments to the corresponding numpy.ndarray.
        Value of all the arguments must be provided.
    aux_states : None or list of numpy.ndarray or dict of str to numpy.ndarray
        Auxiliary states values used as location to compute gradient
        Maps the name of aux_states to the corresponding numpy.ndarray.
        Value of all the auxiliary arguments must be provided.
    eps : float, optional
        epsilon for the finite-difference method

    References
    ---------
    ..[1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """
    for k, v in location.items():
        executor.arg_dict[k][:] = v
    approx_grads = {k:np.zeros(v.shape, dtype=np.float32) for k, v in location.items()}

    executor.forward(is_train=use_forward_train)
    f_x = executor.outputs[0].asnumpy()[0]
    for k, v in location.items():
        old_value = v.copy()
        for i in range(np.prod(v.shape)):
            # inplace update
            v.reshape((np.prod(v.shape), 1))[i] += eps
            # set initial states. Need to set all due to inplace operations
            for key, val in location.items():
                executor.arg_dict[key][:] = val
            if aux_states is not None:
                for key, val in aux_states.items():
                    executor.aux_dict[key][:] = val
            executor.forward(is_train=use_forward_train)
            f_eps = executor.outputs[0].asnumpy()[0]
            approx_grads[k].ravel()[i] = (f_eps - f_x) / eps
            v.reshape((np.prod(v.shape), 1))[i] = old_value.reshape((np.prod(v.shape), 1))[i]

    return approx_grads


def check_numeric_gradient(sym, location, aux_states=None, numeric_eps=1e-4, check_eps=1e-2,
                           grad_nodes=None, use_forward_train=True, ctx=mx.cpu()):
    """Verify an operation by checking backward pass via finite difference method.

    Based on Theano's `theano.gradient.verify_grad` [1]

    Parameters
    ----------
    sym : Symbol
        Symbol containing op to test
    location : list or tuple or dict
        Argument values used as location to compute gradient

        - if type is list of numpy.ndarray
            inner elements should have the same the same order as mxnet.sym.list_arguments().
        - if type is dict of str -> numpy.ndarray
            maps the name of arguments to the corresponding numpy.ndarray.
        *In either case, value of all the arguments must be provided.*
    aux_states : ist or tuple or dict, optional
        The auxiliary states required when generating the executor for the symbol
    numeric_eps : float, optional
        Delta for the finite difference method that approximates the gradient
    check_eps : float, optional
        relative error eps used when comparing numeric grad to symbolic grad
    grad_nodes : None or list or tuple or dict, optional
        Names of the nodes to check gradient on
    use_forward_train : bool
        Whether to use is_train=True when computing the finite-difference
    ctx : Context, optional
        Check the gradient computation on the specified device
    References
    ---------
    ..[1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """

    def random_projection(shape):
        """Get a random weight matrix with not too small elements

        Parameters
        ----------
        shape : list or tuple
        """
        # random_projection should not have elements too small,
        # otherwise too much precision is lost in numerical gradient
        plain = _rng.rand(*shape) + 0.1
        return plain
    location = _parse_location(sym=sym, location=location, ctx=ctx)
    location_npy = {k:v.asnumpy() for k, v in location.items()}
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx)
    if aux_states is not None:
        aux_states_npy = {k:v.asnumpy() for k, v in aux_states.items()}
    else:
        aux_states_npy = None
    if grad_nodes is None:
        grad_nodes = sym.list_arguments()
        grad_req = {k: 'write' for k in grad_nodes}
    elif isinstance(grad_nodes, (list, tuple)):
        grad_nodes = list(grad_nodes)
        grad_req = {k: 'write' for k in grad_nodes}
    elif isinstance(grad_nodes, dict):
        grad_req = grad_nodes.copy()
        grad_nodes = grad_nodes.keys()
    else:
        raise ValueError

    input_shape = {k: v.shape for k, v in location.items()}
    _, out_shape, _ = sym.infer_shape(**input_shape)
    proj = mx.sym.Variable("__random_proj")