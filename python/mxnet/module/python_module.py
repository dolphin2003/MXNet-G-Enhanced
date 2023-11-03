# pylint: disable=too-many-instance-attributes, too-many-arguments
"""Provide some handy classes for user to implement a simple computation module
in Python easily.
"""
import logging

from .base_module import BaseModule
from ..initializer import Uniform
from .. import ndarray as nd

class PythonModule(BaseModule):
    """A convenient module class that implements many of the module APIs as
    empty functions.

    Parameters
    ----------
    data_names : list of str
        Names of the data expected by the module.
    label_names : list of str
        Names of the labels expected by the module. Could be `None` if the
        module does not need labels.
    output_names : list of str
        Names of the outputs.
    """
    def __init__(self, data_names, label_names, output_names, logger=logging):
        super(PythonModule, self).__init__(logger=logger)

        if isinstance(data_names, tuple):
            data_names = list(data_names)
        if isinstance(label_names, tuple):
            label_names = list(label_names)

        self._data_names = data_names
        self._label_names = label_names
        self._output_names = output_names

        self._data_shapes = None
        self._label_shapes = None
        self._output_shapes = None

    ################################################################################
    # Symbol information
    ################################################################################
    @property
    def data_names(self):
        """A list of names for data required by this module."""
        return self._data_names

    @property
    def output_names(self):
        """A list of names for the outputs of this module."""
        return self._output_names

    ################################################################################
    # Input/Output information
    ################################################################################
    @property
    def data_shapes(self):
        """A list of (name, shape) pairs specifying the data inputs to this module."""
        return self._data_shapes

    @property
    def label_shapes(self):
        """A list of (name, shape) pairs specifying the label inputs to this module.
        If this module does not accept labels -- either it is a module without loss
        function, or it is not binded for training, then this should return an empty
        list `[]`.
        """
        return self._label_shapes

    @property
    def output_shapes(self):
        """A list of (name, shape) pairs specifying the outputs of this module."""
        return self._output_shapes

    ################################################################################
    # Parameters of a module
    ################################################################################
    def get_params(self):
        """Get parameters, those are potentially copies of the the actual parameters used
        to do computation on the device.

        Returns
        -------
        `({}, {})`, a pair of empty dict. Subclass should override this method if
        contains parameters.
        """
        return (dict(), dict())

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False):
        """Initialize the parameters and auxiliary states. By default this function
        does nothing. Subclass should override this method if contains parameters.

        Parameters
        ----------
        initializer : Initializer
            Called to initialize parameters if needed.
        arg_params : dict
            If not None, should be a dictionary of existing arg_params. Initialization
            will be copied from that.
        aux_params : dict
            If not None, should be a dictionary of existing aux_params. Initialization
            will be copied from that.
        allow_missing : bool
            If true, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If true, will force re-initialize even if already initialized.
        """
        pass

    def update(self):
        """Update parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch. Currently we do nothing here. Subclass should
        override this method if contains parameters.
        """
        pass

    def update_metric(self, eval_metric, labels):
        """Evaluate and accumulate evaluation metric on outputs of the last forward computation.
        ubclass should override this method if needed.

        Parameters
        ----------
        eval_metric : EvalMetric
        labels : list of NDArray
            Typically `data_batch.label`.
        """
        if self._label_shapes is None:
            # since we do not need labels, we are probably not a module with a loss
            # function or predictions, so just ignore this call
            return

        # by default we expect our outputs are some scores that could be evaluated
        eval_metric.update(labels, self.get_outputs())

    ################################################################################
    # module setup
    ################################################################################
    def bind(self, data_shapes, label_shapes=N