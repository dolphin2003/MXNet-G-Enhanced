import numpy as np
import os
from helper.dataset import pascal_voc
from helper.processing import roidb
from helper.processing.bbox_regression import expand_bbox_regression_targets
from helper.processing.bbox_transform import bbox_pred


def visualize_gt_roidb(imdb, gt_roidb):
    """
    visualize gt roidb
    :param imdb: the imdb to be visualized
    :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    :return: None
    """
    import matplotlib.pyplot as plt
    import skimage.io
    for i in range(len(gt_roidb)):
        im_path = imdb.image_path_from_index(imdb.image_set_index[i])
        im = skimage.io.imread(im_path)
        roi_rec = gt_roidb[i]
        plt.imshow(im)
        for bbox, gt_class, overlap in zip(roi_rec['boxes'], roi_rec['gt_classes'], roi_rec['gt_overlaps']):
            box = plt.Rectangle((bbox[0], bbox[1]),
                                bbox[2] - bbox[0],
                                bbox[3] - bbox[1], fill=False,
                                edgecolor='g', linewidth=3)
            plt.gca().add_patch(box)
            plt.gca().text(bbox[0], bbox[1], imdb.classes[gt_class] + ' {}'.format(overlap[0, gt_class]), color='w')
        plt.show()


def visualize_ss_roidb(imdb, ss_roidb, thresh=0.2):
    """
    visualize ss roidb and gt roidb
    :param imdb: the imdb to be visualized
    :param ss_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        already merged with gt roidb
    :param thresh: only bbox with overlap > thresh will be visualized
    :return: None
    """
    import matplotlib.pyplot as plt
    import skimage.io
    for i in range(len(ss_roidb)):
        im_path = imdb.image_path_from_index(imdb.image_set_index[i])
        im = skimage.io.imread(im_path)
        roi_rec = ss_roidb[i]
        ss_indexes = np.where(roi_rec['gt_classes'] == 0)[0]
        gt_indexes = np.where(roi_rec['gt_classes'] != 0)[0]

        gt_overlaps = roi_rec['gt_overlaps'].toarray()
        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)

        for ss_bbox, ss_class, overlap in zip(roi_rec['boxes'][ss_indexes], max_classes, max_overlaps):
            if overlap < thresh:
                continue
            plt.imshow(im)
            for bbox, gt_class in zip(roi_rec['boxes'][gt_indexes], roi_rec['gt_classes'][gt_indexes]):
                box = plt.Rectangle((bbox[0], bbox[1]),
                                    bbox[2] - bbox[0],
                                    bbox[3] - bbox[1], fill=False,
                                    edgecolor='g', linewidth=3)
                plt.gca().add_patch(box)
                plt.gca().text(bbox[0], bbox[1], imdb.classes[gt_class], color='w')
            ss_box = plt.Rectangle((ss_bbox[0], ss_bbox[1]),
                                   ss_bbox[2] - ss_bbox[0],
                                   ss_bbox[3] - ss_bbox[1], fill=False,
                                   edgecolor='b', linewidth=3)
            plt.gca().add_patch(ss_box)
            plt.gca().text(ss_bbox[0], ss_bbox[1], imdb.classes[ss_class] + ' {}'.format(overlap), color='w')
            plt.show()


def visualize_bbox_regression(imdb, roidb, thresh=0.5):
    """
    visualize the target of bounding box regression
    :param imdb: the imdb to be visualized
    :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        ['image', 'max_classes', 'max_overlaps', 'bbox_targets']
    :param thresh: only bbox with overlap > thresh will be visualized
    :return: None
    """
    import matplotlib.pyplot as plt
    import 