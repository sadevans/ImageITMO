import torch.nn as nn
import numpy as np
import math




def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)



class DepthwiseConv(nn.Sequential):
    """
    Depthwise-Convolution-BatchNormalization-Activation Module
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups,
                 norm_layer, act=None):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=(kernel_size-1)//2, groups=groups, bias=False),
            norm_layer(out_channels),
        ]

        if act is not None:
            layers.append(act())

        super(DepthwiseConv, self).__init__(*layers)


class PointwiseConv(nn.Sequential):
    """
    Pointwise-Convolution-Linear-Activation Module
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 norm_layer, act=None):
        kernel_size = 1
        layers = [
            nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                      padding=(kernel_size-1)//2, bias=False),
            norm_layer(out_channels),
        ]

        if act is not None:
            layers.append(act())

        super(PointwiseConv, self).__init__(*layers)


class ConvBNReLU(nn.Sequential):
    """A Sequence of regular convolution, batch normalization and
    ReLU6 activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidualBottleneck(nn.Module):
    """Inverted Residual Bottleneck Module"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidualBottleneck, self).__init__()

        self.stride = stride
        self.hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []

        if expand_ratio != 1:
            layers.append(PointwiseConv(
                    self.hidden_dim,
                    out_channels,
                    kernel=1,
                    stride=1,
                    norm_layer=nn.BatchNorm2d,
                    act=nn.ReLU6,
                ))
        layers.extend([
            DepthwiseConv(
                    self.hidden_dim, self.hidden_dim, 3,
                    stride=stride, groups=self.hidden_dim,
                    norm_layer=nn.BatchNorm2d,
                    act=nn.ReLU6,
                ),
            PointwiseConv(
                self.hidden_dim,
                out_channels,
                kernel=1,
                stride=1,
                norm_layer=nn.BatchNorm2d,
            ),
        ])

        self.conv = nn.Sequential(*layers)


    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, input_size, num_classes=1000, width_mult=1):
        super(MobileNetV2, self).__init__()

        block = InvertedResidualBottleneck
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t: expansion factor, c: output channels, n: number of blocks, s: stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0, "Input size must be divisible by 32"

        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel

        self.features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Linear(self.last_channel, num_classes)

        self._init_weights()

    def _init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
                mod.weight.data.normal_(0, math.sqrt(2. / n))
                if mod.bias is not None:
                    mod.bias.data.zero_()
            elif isinstance(mod, nn.BatchNorm2d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.Linear):
                n = mod.weight.size(1)
                mod.weight.data.normal_(0, 0.01)
                mod.bias.data.zero_()
