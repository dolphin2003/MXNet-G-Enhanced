from common import *

import random
import time

import ctypes
import numpy
import sys
import re

c_float_ptr = ctypes.POINTER(ctypes.c_float)
c_int_ptr = ctypes.POINTER(ctypes.c_int)
c_void_p = ctypes.c_void_p
c_int = ctypes.c_int
c_c