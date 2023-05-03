# pylint: skip-file
import sys
import os
# code to automatically download dataset
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path = [os.path.join(curr_path, "../autoencoder")] + sys.path
import mxnet as mx
import numpy as np
import data
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import model
from autoencoder import AutoEncoderModel
from solver import Solver, Monitor
import logging

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

class DECModel(model.MXModel):
    class DECLoss(mx.operator.NumpyOp):
        def __init__(self, num_centers, alpha):
            super(DECModel.DECLoss, self).__init__(need_top_grad=False)
            self.num_centers = num_centers
            self.alpha = alpha

        def forward(self, in_data, out_data):
            z = in_data[0]
            mu = in_data[1]
            q = out_data[0]
            self.mask = 1.0/(1.0+cdist(z, mu)**2/self.alpha)
            q[:] = self.mask**((self.alpha+1.0)/2.0)
            q[:] = (q.T/q.sum(axis=1)).T

        def backward(self, out_grad, in_data, out_data, in_grad):
            q = out_data[0]
            z = in_data[0]
            mu = in_data[1]
            p = in_data[2]
            dz = in_grad[0]
            dmu = in_grad[1]
            self.mask *= (self.alpha+1.0)/self.alpha*(p-q)
            dz[:] = (z.T*self.mask.sum(axis=1)).T - self.mask.dot(mu)
            dmu[:] = (mu.T*self.mask.sum(axis=0)).T - self.mask.T.dot(z)

        def infer_shape(self, in_shape):
            assert len(in_shape) == 3
            assert len(in_shape[