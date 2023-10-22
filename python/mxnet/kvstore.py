# coding: utf-8
""" Key value store interface of MXNet for parameter synchronization."""
from __future__ import absolute_import

import ctypes
import pickle
from .ndarray import NDArray
from .base import _LIB
from .base import 