# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, random
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from lstm import lstm_unroll

from io import BytesIO
from captcha.image import ImageCaptcha
import cv2, random

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def gen_rand():
    buf = ""
    max_len = random.randint(3,4)
    for i in range(max_len):
        buf += str(random.randint(0,9))
    return buf

def get_label(buf):
    ret = np.zeros(4)
    for i in range(len(buf)):
        ret[i] = 1 + int(buf[i])
    if len(buf) == 3:
        ret[3] = 0
    return ret

class OCRIter(mx.io.DataIter):
    def __init__(self, count, batch_size, num_label, init_states):
        super(OCRIter, self).__init__()
        self.captcha = ImageCaptcha(fonts=['./data/Xerox.ttf'])
        self.batch_size = batch_size
        self.count = count
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, 2400))] + init_states
        self.provide_label = [('label', (self.batch_size, 4))]

    def __iter__(self):
        print 'iter'
        init_state_names = [x[0] for x in self.init_states]
        for k in range(self.count):
            data = []
            label = []
            for i in range(self.batch_size):
                num = gen_rand()
                img = self.captcha.generate(num)
                img = np.fromstring(img.getvalue(), dtype='uint8')
                img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (80, 30))
                img = img.transpose(1, 0)
                img = img.reshape((80 * 30))
                img = np.multiply(img, 1/255.0)
                data.append(img)
                label.append(get_label(num))

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']
            
            
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

BATCH_SIZE = 32
SEQ_LENGTH = 80

def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret

def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    re