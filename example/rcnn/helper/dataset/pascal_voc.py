"""
Pascal VOC database
This class loads ground truth notations from standard Pascal VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.
"""

import os
import numpy as np
import scipy.sparse
import scipy.io
import cPickle
from imdb import IMDB
from voc_eval import voc_eval
from helper.processing.bbox_process import unique_boxes, filter_small_boxes


class PascalVOC(IMDB):
    def __init__(self, image_set, year, root_path, devkit_path):
        """
        fill basic information to initialize imdb
        :param image_set: train or val or trainval
        :param year: 2007, 2010, 2012
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        super(PascalVOC, self).__init__('voc_' + year + '_' + image_set)  # set self.name
        self.image_set = image_set
        self.year = year
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'VOC' + year)

        self.classes = ['__background__',  # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
        self.num_classes = 21
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'm