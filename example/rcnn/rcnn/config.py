import numpy as np
from easydict import EasyDict as edict

config = edict()

# image processing config
config.EPS = 1e-14
config.PIXEL_MEANS = np.array([[[123.68, 116.779, 103.939]]])
config.SCALES = (600, )  # single scale training and testing
config.MAX_SIZE = 1000

# nms config
config.USE_GPU_NMS = True
config.GPU_ID = 0

config.TRAIN = edict()

# R-CNN and RP