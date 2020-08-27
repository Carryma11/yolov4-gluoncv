from mxnet.gluon import nn
import mxnet as mx
from tools.mx_utils import *
from tools.yolo_layer import YoloLayer
from mxnet import autograd

class Mish(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = x * (F.tanh(F.Activation(data=x,act_type='softrelu')))
        return x


""" CBM:Conv2D->BatchNorm->Mish """


def CBM(num_channels, kernel_size, strides):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    Conv_Bn_Mish = nn.HybridSequential()
    with Conv_Bn_Mish.name_scope():
        Conv_Bn_Mish.add(nn.Conv2D(num_channels, kernel_size=kernel_size,
                                   strides=strides, padding=pad, use_bias=False),
                         nn.BatchNorm(), Mish())
    return Conv_Bn_Mish


""" CBL:Conv2D->BatchNorm->LeakyReLu """


def CBL(num_channels, kernel_size, strides):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    Conv_Bn_Leaky = nn.HybridSequential()
    with Conv_Bn_Leaky.name_scope():
        Conv_Bn_Leaky.add(nn.Conv2D(num_channels, kernel_size=kernel_size,
                                    strides=strides, padding=pad, use_bias=False),
                          nn.BatchNorm(), nn.LeakyReLU(0.1))
    return Conv_Bn_Leaky


""" Make cbls used for Neck. """


def make_cbls(num_filters, filter_list):
    cbls = nn.HybridSequential()
    for n in range(num_filters):
        if n % 2 == 0:
            cbls.add(CBL(filter_list[0], 1, 1))
        else:
            cbls.add(CBL(filter_list[1], 3, 1))
    return cbls


""" SPP structure """


class SpatialPyramidPooling(nn.HybridBlock):
    def __init__(self):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpool1 = nn.MaxPool2D(pool_size=5, strides=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2D(pool_size=9, strides=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2D(pool_size=13, strides=1, padding=13 // 2)

    def hybrid_forward(self, F, x, *args, **kwargs):
        spp1 = self.maxpool1(x)
        spp2 = self.maxpool2(x)
        spp3 = self.maxpool3(x)
        features = F.concat(*[spp1, spp2, spp3, x], dim=1)

        return features


class Upsample(nn.HybridBlock):
    def __init__(self, num_channels):
        super(Upsample, self).__init__()
        self.cbl = CBL(num_channels, 1, 1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x1 = self.cbl(x)
        up = F.UpSampling(x1, scale=2, sample_type='nearest')
        return up


"""each of Resunit consists of two Conv_Bn_Mish layers."""


class ResUnit(nn.HybridBlock):
    def __init__(self, num_channels, hid_channels=None):
        super(ResUnit, self).__init__()

        if hid_channels is None:
            hid_channels = num_channels
        self.cbm1 = CBM(hid_channels, 1, 1)
        self.cbm2 = CBM(num_channels, 3, 1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        h = self.cbm2(self.cbm1(x))
        return x + h


def ResBlock(num_channels, num_units, first_block=False):
    blk = nn.HybridSequential()
    for i in range(num_units):
        if i == 0 and first_block:
            blk.add(ResUnit(num_channels, num_channels // 2))
        else:
            blk.add(ResUnit(num_channels))

    return blk


class CSPX(nn.HybridBlock):
    def __init__(self, num_channels, num_units, first_cspx=False):
        super(CSPX, self).__init__()

        self.downsample_conv = CBM(num_channels, 3, 2)
        if first_cspx:
            self.cbm1 = CBM(num_channels, 1, 1)
            self.cspx = self._get_cspx_first(num_channels, num_units)
            self.cbm2 = CBM(num_channels, 1, 1)
        else:
            self.cbm1 = CBM(num_channels // 2, 1, 1)
            self.cpsx = self._get_cspx_normal(num_channels, num_units)
            self.cbm2 = CBM(num_channels, 1, 1)

    def _get_cspx_first(self, num_channels, num_units):
        self.cspx = nn.HybridSequential()
        self.cspx.add(CBM(num_channels, 1, 1),
                      ResBlock(num_channels, num_units, first_block=True),
                      CBM(num_channels, 1, 1))
        return self.cspx

    def _get_cspx_normal(self, num_channels, num_units):
        self.cspx = nn.HybridSequential()
        self.cspx.add(CBM(num_channels // 2, 1, 1),
                      ResBlock(num_channels // 2, num_units),
                      CBM(num_channels // 2, 1, 1))
        return self.cspx

    def hybrid_forward(self, F, x, *args, **kwargs):
        x1 = self.downsample_conv(x)
        x2 = self.cbm1(x1)
        x3 = self.cspx(x1)
        x4 = F.concat(x2, x3, dim=1)
        x5 = self.cbm2(x4)
        return x5


class CSPDDarkNet53(nn.HybridBlock):
    def __init__(self):
        super(CSPDDarkNet53, self).__init__()
        self.conv1 = CBM(32, 3, 1)
        self.channel_list = [64, 128, 256, 512, 1024]
        self.unit_list = [1, 2, 8, 8, 4]
        self.CSPD1 = CSPX(self.channel_list[0], self.unit_list[0], first_cspx=True)
        self.CSPD2 = CSPX(self.channel_list[1], self.unit_list[1])
        self.CSPD3 = CSPX(self.channel_list[2], self.unit_list[2])
        self.CSPD4 = CSPX(self.channel_list[3], self.unit_list[3])
        self.CSPD5 = CSPX(self.channel_list[4], self.unit_list[4])

    def hybrid_forward(self, F, x, *args, **kwargs):
        x1 = self.conv1(x)
        x2 = self.CSPD1(x1)
        x3 = self.CSPD2(x2)
        out0 = self.CSPD3(x3)
        out1 = self.CSPD4(out0)
        out2 = self.CSPD5(out1)

        return out0, out1, out2


class DarkNetV4(nn.HybridBlock):
    def __init__(self, classes=1000, **kwargs):
        super(DarkNetV4, self).__init__(**kwargs)
        self.channel_list = [64, 128, 256, 512, 1024]
        self.unit_list = [1, 2, 8, 8, 4]
        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(CBM(32, 3, 1))
            self.features.add(CSPX(self.channel_list[0], self.unit_list[0], first_cspx=True))
            self.features.add(CSPX(self.channel_list[1], self.unit_list[1]))
            self.features.add(CSPX(self.channel_list[2], self.unit_list[2]))
            self.features.add(CSPX(self.channel_list[3], self.unit_list[3]))
            self.features.add(CSPX(self.channel_list[4], self.unit_list[4]))

        self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = F.Pooling(x, kernel=(7, 7), global_pool=True, pool_type='avg')
        return self.output(x)


def get_darknet_base(pretrained=False, ctx=mx.cpu(), **kwargs):
    net = DarkNetV4(**kwargs)
    if pretrained:
        net.load_parameters('model_weight/CSPDDarkNet53', ctx=ctx)
    return net



class Neck(nn.HybridBlock):
    def __init__(self):
        super(Neck, self).__init__()

        self.make_three_cbl1 = make_cbls(3, [512, 1024])
        self.spp = SpatialPyramidPooling()
        # 这里的输入in_units不一样,必须要创建两个
        self.make_three_cbl2 = make_cbls(3, [512, 1024])
        self.upsample1 = Upsample(256)

        self.cbl_for_P4 = CBL(256, 1, 1)
        self.make_five_cbl1 = make_cbls(5, [256, 512])

        self.upsample2 = Upsample(128)
        self.cbl_for_P3 = CBL(128, 1, 1)
        self.make_five_cbl2 = make_cbls(5, [128, 256])

        self.down_sample1 = CBL(256, 3, 2)
        self.down_sample2 = CBL(512, 3, 2)
        self.make_five_cbl3 = make_cbls(5, [256, 512])
        self.make_five_cbl4 = make_cbls(5, [512, 1024])

        self.head_0 = CBL(256, 3, 1)
        self.head_1 = CBL(512, 3, 1)
        self.head_2 = CBL(1024, 3, 1)

    def hybrid_forward(self, F, x_big, x_mid, x_tiny, *args, **kwargs):
        P5 = self.make_three_cbl1(x_tiny)
        P5 = self.spp(P5)
        P5 = self.make_three_cbl2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.cbl_for_P4(x_mid)
        P4 = F.concat(P4, P5_upsample, dim=1)
        P4 = self.make_five_cbl1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.cbl_for_P3(x_big)
        P3 = F.concat(P3, P4_upsample, dim=1)
        P3 = self.make_five_cbl2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = F.concat(P3_downsample, P4, dim=1)
        P4 = self.make_five_cbl3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = F.concat(P4_downsample, P5, dim=1)
        P5 = self.make_five_cbl4(P5)

        P3 = self.head_0(P3)
        P4 = self.head_1(P4)
        P5 = self.head_2(P5)

        return P3, P4, P5


def yolo_head(filters_list):
    pre = nn.HybridSequential()
    pre.add(CBL(filters_list[0], 3, 1),
            nn.Conv2D(filters_list[1], 1, 1))

    return pre


class Yolov4Body(nn.HybridBlock):
    def __init__(self, num_classes):
        super(Yolov4Body, self).__init__()
        # backbone
        self.backbone = CSPDDarkNet53()
        # Neck
        self.Neck = Neck()
        # three anchor boxes on each feature map
        # take VOC type for example:3 * (5 + 20) = 3 * (4 + 1 + 20) = 75
        output_channel = 3 * (5 + num_classes)
        # yolo_Head
        self.yolo_head3 = nn.Conv2D(output_channel, 1, 1)
        self.yolo_head2 = nn.Conv2D(output_channel, 1, 1)
        self.yolo_head1 = nn.Conv2D(output_channel, 1, 1)
        # inference
        self.yolo1 = YoloLayer(
            anchor_mask=[0, 1, 2], num_classes=num_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9, stride=8)
        self.yolo2 = YoloLayer(
            anchor_mask=[3, 4, 5], num_classes=num_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9, stride=16)
        self.yolo3 = YoloLayer(
            anchor_mask=[6, 7, 8], num_classes=num_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9, stride=32)

    def hybrid_forward(self, F, x, *args, **kwargs):
        feat0, feat1, feat2 = self.backbone(x)
        feat_top, feat_mid, feat_bottom = self.Neck(feat0, feat1, feat2)
        out_top = self.yolo_head3(feat_top)
        out_mid = self.yolo_head2(feat_mid)
        out_bottom = self.yolo_head1(feat_bottom)
        y1 = self.yolo1(out_top)
        y2 = self.yolo2(out_mid)
        y3 = self.yolo3(out_bottom)
        if autograd.is_training():
            return y1, y2, y3
        else:
            return get_region_boxes([y1, y2, y3])



from mxnet import nd
num_classes = 80
net = Yolov4Body(num_classes=80)
#print(net)
net.initialize(ctx=mx.gpu())
#net.hybridize()
X = nd.zeros(shape=(1, 3, 608, 608),ctx=mx.gpu(0))

print(net(X)[0].shape,net(X)[1].shape)
