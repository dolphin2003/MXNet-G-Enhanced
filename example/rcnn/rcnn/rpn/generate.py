import mxnet as mx
import numpy as np
import os
import cPickle


class Detector(object):
    def __init__(self, symbol, ctx=None,
                 arg_params=None, aux_params=None):
        self.symbol = symbol
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        self.executor = None
        self.arg_params = arg_params
        self.aux_params = aux_params

    def im_detect(self, im, im_info):
        """
        perform detection of im, im_info
        :param im: numpy.ndarray [b, c, h, w]
        :param im_info: numpy.ndarray [b, 3]
        :return: boxes [b, 5], scores [b,]
        """
        self.arg_params['data'] = mx.nd.array(im, self.ctx)
        self.arg_params['im_info'] = mx.nd.array(im_info, self.ctx)
        arg_shapes, out_shapes, aux_shapes = \
            self.symbol.infer_shape(data=self.arg_params['data'].shape, im_info=self.arg_params['im_info'].shape)
        aux_names = self.symbol.list_auxiliary_states()
        self.aux_params = {k: mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}
        self.executor = self.symbol.bind(self.ctx, self.arg_params, args_grad=None,
                                         grad_req='null', aux_states=self.aux_params)
        output_dict = {name: nd for name, nd in zip(self.symbol.list_outputs(), self.executor.outputs)}

        self.executor.forward(is_train=False)
        boxes = output_dict['rois_output'].asnumpy()
        scores = output_dict['rois_score'].asnumpy()

        return boxes, scores


def generate_detections(detector, test_data, imdb, vis=False):
    """
    Generate detections results using RPN.
    :param detector: Detector
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :return: list of detected boxes
    """
    assert not test_data.shuffle

    i = 0
    imdb_boxes = 