from mxnet.gluon import nn
import mxnet as mx
import numpy as np
from tools.mx_utils import *
from mxnet import autograd
ctx = mx.gpu()


def yolo_forward(output, conf_thresh, num_classes, anchors, num_anchors, scale_x_y, only_objectness=1,
                 validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    batch = output.shape[0]
    H = output.shape(2)
    W = output.shape(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)

        bxy_list.append(output[:, begin: begin + 2])
        bwh_list.append(output[:, begin + 2: begin + 4])
        det_confs_list.append(output[:, begin + 4: begin + 5])
        cls_confs_list.append(output[:, begin + 5: end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = nd.concat(*bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = nd.concat(*bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = nd.concat(*det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.reshape(batch, num_anchors * H * W)

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = nd.concat(*cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.reshape(batch, num_anchors, num_classes, H * W)
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, num_classes)

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = nd.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = nd.exp(bwh)
    det_confs = nd.sigmoid(det_confs)
    cls_confs = nd.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are nd related)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0), axis=0),
                            axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1), axis=0),
                            axis=0)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii: ii + 1] + nd.array(grid_x, ctx=ctx, dtype='float32')
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1: ii + 2] + nd.array(grid_y, ctx=ctx, dtype='float32')
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii: ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1: ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)

    ########################################
    #   Figure out bboxes from slices     #
    ########################################

    # Shape: [batch, num_anchors, H, W]
    bx = nd.concat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = nd.concat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = nd.concat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = nd.concat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = nd.concat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = nd.concat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= W
    by_bh /= H

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx_bw[:, :num_anchors].reshape(batch, num_anchors * H * W, 1)
    by = by_bh[:, :num_anchors].reshape(batch, num_anchors * H * W, 1)
    bw = bx_bw[:, num_anchors:].reshape(batch, num_anchors * H * W, 1)
    bh = by_bh[:, num_anchors:].reshape(batch, num_anchors * H * W, 1)

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    boxes = nd.concat((bx1, by1, bx2, by2), dim=2).reshape(batch, num_anchors * H * W, 1, 4)
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.reshape(batch, num_anchors * H * W, 1)
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return boxes, confs


def yolo_forward_dynamic(output, conf_thresh, num_classes, anchors, num_anchors, scale_x_y, only_objectness=1,
                         validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    # batch = output.size[0]
    # H = output.size(2)
    # W = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)

        bxy_list.append(output[:, begin: begin + 2])
        bwh_list.append(output[:, begin + 2: begin + 4])
        det_confs_list.append(output[:, begin + 4: begin + 5])
        cls_confs_list.append(output[:, begin + 5: end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = nd.concat(*[bxy_list[i]for i in range(num_anchors)], dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = nd.concat(*[bwh_list[i]for i in range(num_anchors)], dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = nd.concat(*[det_confs_list[i]for i in range(num_anchors)], dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.reshape(output.shape[0], num_anchors * output.shape[2] * output.shape[3])

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = nd.concat(*[cls_confs_list[i]for i in range(num_anchors)], dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.reshape(output.shape[0], num_anchors, num_classes, output.shape[2] * output.shape[3])
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = nd.transpose(cls_confs, axes=(0, 1, 3, 2))
    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = nd.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = nd.exp(bwh)
    det_confs = nd.sigmoid(det_confs)
    cls_confs = nd.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are nd related)
    grid_x = np.expand_dims(np.expand_dims(
        np.expand_dims(np.linspace(0, output.shape[3] - 1, output.shape[3]), axis=0).repeat(output.shape[2], 0), axis=0),
        axis=0)
    grid_y = np.expand_dims(np.expand_dims(
        np.expand_dims(np.linspace(0, output.shape[2] - 1, output.shape[2]), axis=1).repeat(output.shape[3], 1), axis=0),
        axis=0)
    # grid_x = nd.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = nd.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)
    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii: ii + 1] + nd.array(grid_x, ctx=ctx,dtype='float32')
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1: ii + 2] + nd.array(grid_y, ctx=ctx,dtype='float32')
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii: ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1: ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)

    ########################################
    #   Figure out bboxes from slices     #
    ########################################

    # Shape: [batch, num_anchors, H, W]
    bx = nd.concat(*[bx_list[i]for i in range(num_anchors)], dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = nd.concat(*[by_list[i]for i in range(num_anchors)], dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = nd.concat(*[bw_list[i]for i in range(num_anchors)], dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = nd.concat(*[bh_list[i]for i in range(num_anchors)], dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = nd.concat(bx, bw, dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = nd.concat(by, bh, dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= output.shape[3]
    by_bh /= output.shape[2]

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx_bw[:, :num_anchors].reshape(output.shape[0], num_anchors * output.shape[2] * output.shape[3], 1)
    by = by_bh[:, :num_anchors].reshape(output.shape[0], num_anchors * output.shape[2] * output.shape[3], 1)
    bw = bx_bw[:, num_anchors:].reshape(output.shape[0], num_anchors * output.shape[2] * output.shape[3], 1)
    bh = by_bh[:, num_anchors:].reshape(output.shape[0], num_anchors * output.shape[2] * output.shape[3], 1)

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    boxes = nd.concat(bx1, by1, bx2, by2, dim=2).reshape(output.shape[0], num_anchors * output.shape[2] * output.shape[3],
                                                        1, 4)
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]
    cls_confs = cls_confs.reshape(output.shape[0], num_anchors * output.shape[2] * output.shape[3], -1)
    det_confs = det_confs.reshape(output.shape[0], num_anchors * output.shape[2] * output.shape[3], 1)
    confs = nd.broadcast_mul(cls_confs,det_confs)

    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return boxes, confs


class YoloLayer(nn.HybridBlock):
    ''' Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    '''

    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, stride=32, model_out=False):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out

    def hybrid_forward(self, F, output, target=None):
        if autograd.is_training():
            return output
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        return yolo_forward_dynamic(output, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask),
                                    scale_x_y=self.scale_x_y)



'''
    x=nd.zeros(shape=(1,256,76,76),ctx=ctx)
    masked_anchors = []
    anchor_mask =[0,1,2]
    num_anchors =9
    stride = 8
    anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    anchor_step = len(anchors) // num_anchors
    for m in anchor_mask:
        masked_anchors += anchors[m * anchor_step:(m + 1) * anchor_step]
    masked_anchors = [anchor / 8 for anchor in masked_anchors]

    yolo1 =yolo_forward_dynamic(x,conf_thresh=0.6, num_classes=80, anchors=masked_anchors,num_anchors=len(anchor_mask),scale_x_y=1)
'''