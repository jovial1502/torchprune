import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, Iterable, defaultdict

from .yolo_layer import *

class Yolov3Tiny(nn.Module):
    
    def __init__(self, num_classes, use_wrong_previous_anchors=False):
        super().__init__()

        self.num_classes = num_classes
        self.return_out_boxes = False
        self.skip_backbone = False
        self.input_channels=3

        self.layers_list =  OrderedDict([
            ('0_convbatch',     ConvBN(self.input_channels, 16, 3, 1, 1)),
            ('1_max',           nn.MaxPool2d(2, 2)),
            ('2_convbatch',     ConvBN(16, 32, 3, 1, 1)),
            ('3_max',           nn.MaxPool2d(2, 2)),
            ('4_convbatch',     ConvBN(32, 64, 3, 1, 1)),
            ('5_max',           nn.MaxPool2d(2, 2)),
            ('6_convbatch',     ConvBN(64, 128, 3, 1, 1)),
            ('7_max',           nn.MaxPool2d(2, 2)),
            ('8_convbatch',     ConvBN(128, 256, 3, 1, 1)),
            ('9_max',           nn.MaxPool2d(2, 2)),
            # ('9_max',           nn.MaxPool2d(2, 2, ceil_mode=True)), 
            ('10_convbatch',    ConvBN(256, 512, 3, 1, 1)),
            ('11_max',          MaxPoolStride1()),
            ('12_convbatch',    ConvBN(512, 1024, 3, 1, 1)),
            ('13_convbatch',    ConvBN(1024, 256, 1, 1, 0)), # padding = kernel_size-1//2
        ])
        self.layers = nn.Sequential(self.layers_list)
        self.idx = 9

        self.backbone = self.layers

        anchors_per_region = 3
        self.yolo_0_pre = nn.Sequential(OrderedDict([
            ('14_convbatch',    ConvBN(256, 512, 3, 1, 1)),
            ('15_conv',         nn.Conv2d(512, anchors_per_region*(5+self.num_classes), 1, 1, 0)),
            # ('16_yolo',         YoloLayer()),
        ]))
        self.yolo_0 = YoloLayer(anchors=[(81.,82.), (135.,169.), (344.,319.)], stride=32, num_classes=num_classes)

        self.up_1 = nn.Sequential(OrderedDict([
            ('17_convbatch',    ConvBN(256, 128, 1, 1, 0)),
            ('18_upsample',     Upsample(2)),
        ]))

        self.yolo_1_pre = nn.Sequential(OrderedDict([
            ('19_convbatch',    ConvBN(128+256, 256, 3, 1, 1)),
            ('20_conv',         nn.Conv2d(256, anchors_per_region*(5+self.num_classes), 1, 1, 0)),
            # ('21_yolo',         YoloLayer()),
        ]))
        
        # Tiny yolo weights were originally trained using wrong anchor mask
        # https://github.com/pjreddie/darknet/commit/f86901f6177dfc6116360a13cc06ab680e0c86b0#diff-2b0e16f442a744897f1606ff1a0f99d3L175
        if use_wrong_previous_anchors:
            yolo_1_anchors = [(23.,27.),  (37.,58.),  (81.,82.)]
        else: 
            yolo_1_anchors = [(10.,14.),  (23.,27.),  (37.,58.)]

        self.yolo_1 = YoloLayer(anchors=yolo_1_anchors, stride=16.0, num_classes=num_classes)

    def get_loss_layers(self):
        return [self.yolo_0, self.yolo_1]

    def forward(self, x):
        x_b_0 = self.layers[:self.idx](x)
        x_b_full = self.layers[self.idx:](x_b_0)
        
        # x_b_0, x_b_full = xb[0], xb[1]
        y0 = self.yolo_0_pre(x_b_full)

        x_up = self.up_1(x_b_full)
        x_up = torch.cat((x_up, x_b_0), 1)
        y1 = self.yolo_1_pre(x_up)

        return y1


###################################################################
## Backbone and helper modules

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x


# class Yolov3TinyBackbone(nn.Module):
#     def __init__(self, input_channels=3):
#         super().__init__()
#         self.layers_list =  OrderedDict([
#             ('0_convbatch',     ConvBN(input_channels, 16, 3, 1, 1)),
#             ('1_max',           nn.MaxPool2d(2, 2)),
#             ('2_convbatch',     ConvBN(16, 32, 3, 1, 1)),
#             ('3_max',           nn.MaxPool2d(2, 2)),
#             ('4_convbatch',     ConvBN(32, 64, 3, 1, 1)),
#             ('5_max',           nn.MaxPool2d(2, 2)),
#             ('6_convbatch',     ConvBN(64, 128, 3, 1, 1)),
#             ('7_max',           nn.MaxPool2d(2, 2)),
#             ('8_convbatch',     ConvBN(128, 256, 3, 1, 1)),
#             ('9_max',           nn.MaxPool2d(2, 2)),
#             # ('9_max',           nn.MaxPool2d(2, 2, ceil_mode=True)), 
#             ('10_convbatch',    ConvBN(256, 512, 3, 1, 1)),
#             ('11_max',          MaxPoolStride1()),
#             ('12_convbatch',    ConvBN(512, 1024, 3, 1, 1)),
#             ('13_convbatch',    ConvBN(1024, 256, 1, 1, 0)), # padding = kernel_size-1//2
#         ])
#         self.layers = nn.Sequential(self.layers_list)
#         self.idx = 9
    
#     def forward(self, x):
#         x_b_0 = self.layers[:self.idx](x)
#         x_b_full = self.layers[self.idx:](x_b_0)
#         return x_b_0, x_b_full   

class ConvBN(nn.Module):
    "convolutional layer then batchnorm"

    def __init__(self, ch_in, ch_out, kernel_size = 3, stride=1, padding=None):
        super().__init__()
        if padding is None: padding = (kernel_size - 1) // 2 # we should never need to set padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x): return self.relu(self.bn(self.conv(x)))


class Upsample(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
    def forward(self, x):
        assert(x.data.dim() == 4)
        return nn.Upsample(scale_factor=self.stride, mode='nearest')(x)      

def yolov3tiny(num_classes):
    """Return a fully-initialized CNN60K."""
    return Yolov3Tiny(num_classes)           