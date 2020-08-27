
from __future__ import absolute_import
from mxnet import gluon
from mxnet import nd
import math
from mxnet.gluon.loss import Loss
__all__ = ['YOLOV4Loss']

def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: NDarray, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: NDarray, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: NDarray, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = nd.max(b1_mins, b2_mins)
    intersect_maxes = nd.min(b1_maxes, b2_maxes)
    intersect_wh = nd.max(intersect_maxes - intersect_mins, nd.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / nd.clip(union_area, a_min=1e-6)

    # 计算中心的差距
    center_distance = nd.sum(nd.power((b1_xy - b2_xy), 2), axis=-1)

    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = nd.min(b1_mins, b2_mins)
    enclose_maxes = nd.max(b1_maxes, b2_maxes)
    enclose_wh = nd.max(enclose_maxes - enclose_mins, nd.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = nd.sum(nd.power(enclose_wh, 2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / nd.clip(enclose_diagonal, a_min=1e-6)

    v = (4 / (math.pi ** 2)) * nd.power((nd.arctan(b1_wh[..., 0] / nd.clip(b1_wh[..., 1], min=1e-6)) - nd.arctan(
        b2_wh[..., 0] / nd.clip(b2_wh[..., 1], a_min=1e-6))), 2)
    alpha = v / nd.clip((1.0 - iou + v), a_max=1e-6)
    ciou = ciou - alpha * v
    return ciou


class YOLOV4Loss(Loss):
    """Losses of YOLO v4.

    Parameters
    ----------
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.

    """
    def __init__(self, batch_axis=0, weight=None, **kwargs):
        super(YOLOV4Loss, self).__init__(weight, batch_axis, **kwargs)
        self._sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self._l1_loss = gluon.loss.L1Loss()

    def hybrid_forward(self, F, objness, box_centers, box_scales, cls_preds,
                       objness_t, center_t, scale_t, weight_t, class_t, class_mask):
        """Compute YOLOv4 losses.

        Parameters
        ----------
        objness : mxnet.nd.NDArray
            Predicted objectness (B, N), range (0, 1).
        box_centers : mxnet.nd.NDArray
            Predicted box centers (x, y) (B, N, 2), range (0, 1).
        box_scales : mxnet.nd.NDArray
            Predicted box scales (width, height) (B, N, 2).
        cls_preds : mxnet.nd.NDArray
            Predicted class predictions (B, N, num_class), range (0, 1).
        objness_t : mxnet.nd.NDArray
            Objectness target, (B, N), 0 for negative 1 for positive, -1 for ignore.
        center_t : mxnet.nd.NDArray
            Center (x, y) targets (B, N, 2).
        scale_t : mxnet.nd.NDArray
            Scale (width, height) targets (B, N, 2).
        weight_t : mxnet.nd.NDArray
            Loss Multipliers for center and scale targets (B, N, 2).
        class_t : mxnet.nd.NDArray
            Class targets (B, N, num_class).
            It's relaxed one-hot vector, i.e., (1, 0, 1, 0, 0).
            It can contain more than one positive class.
        class_mask : mxnet.nd.NDArray
            0 or 1 mask array to mask out ignored samples (B, N, num_class).

        Returns
        -------
        tuple of NDArrays
            obj_loss: sum of objectness logistic loss
            center_loss: sum of box center logistic regression loss
            scale_loss: sum of box scale l1 loss
            cls_loss: sum of per class logistic loss

        """
        # compute some normalization count, except batch-size
        denorm = F.cast(
            F.shape_array(objness_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        weight_t = F.broadcast_mul(weight_t, objness_t)
        hard_objness_t = F.where(objness_t > 0, F.ones_like(objness_t), objness_t)
        new_objness_mask = F.where(objness_t > 0, objness_t, objness_t >= 0)
        obj_loss = F.broadcast_mul(
            self._sigmoid_ce(objness, hard_objness_t, new_objness_mask), denorm)
        center_loss = F.broadcast_mul(self._sigmoid_ce(box_centers, center_t, weight_t), denorm * 2)
        scale_loss = F.broadcast_mul(self._l1_loss(box_scales, scale_t, weight_t), denorm * 2)
        denorm_class = F.cast(
            F.shape_array(class_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        class_mask = F.broadcast_mul(class_mask, objness_t)
        cls_loss = F.broadcast_mul(self._sigmoid_ce(cls_preds, class_t, class_mask), denorm_class)
        return obj_loss, center_loss, scale_loss, cls_loss
