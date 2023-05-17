# pylint:skip-file
import sys, random, time, math
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
from nce import *
from operator import itemgetter
from optparse import OptionParser

LSTMSta