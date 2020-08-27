import sys
import os
import time
import math
from mxnet import nd
import numpy as np

from tools import utils


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = nd.min(boxes1[0], boxes2[0])
        Mx = nd.max(boxes1[2], boxes2[2])
        my = nd.min(boxes1[1], boxes2[1])
        My = nd.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = nd.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = nd.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = nd.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = nd.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def get_region_boxes(boxes_and_confs):
    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = nd.concat(*[boxes_list[i]for i,_ in enumerate(boxes_list)], dim=1)
    confs = nd.concat(*[confs_list[i]for i,_ in enumerate(confs_list)], dim=1)

    return [boxes, confs]

'''
def convert2cpu(gpu_matrix):
    return nd.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return nd.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)
'''

def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = nd.from_numpy(img.transpose(2, 0, 1)).\
        img = nd.broadcast_div(img,255.0)
        img.expend_dims(axis=0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = nd.from_numpy(img.transpose(0, 3, 1, 2))
        img = nd.broadcast_div(img,255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()
    img = nd.autograd.Variable(img)

    t1 = time.time()

    output = model(img)

    t2 = time.time()

    print('-----------------------------------')
    print('           Preprocess : %f' % (t1 - t0))
    print('      Model Inference : %f' % (t2 - t1))
    print('-----------------------------------')

    return utils.post_processing(img, conf_thresh, nms_thresh, output)

