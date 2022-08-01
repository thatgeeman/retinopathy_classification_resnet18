from collections import OrderedDict

import torch
from fastcore.basics import store_attr
from fastcore.imports import noop
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ConvBnAct(nn.Module):
    """Calls `Conv2D`, `BatchNorm2d` and `act_fn` in sequence.
    `noop` with `bn=False`, `act=False`.
    """

    def __init__(self, in_ch=3, out_ch=64, k=3, s=1, p=0, d=1, bn=True, act=True, bias=False):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, d, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch) if bn else noop
        self.act_fn = nn.ReLU(inplace=True) if act else noop

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act_fn(x)


class BasicResBlock(nn.Module):
    """
    Basic Residual block (no 1x1conv).
    """

    def __init__(self, in_ch, out_ch):
        super(BasicResBlock, self).__init__()
        assert in_ch == out_ch, "Regular block should have in_ch==out_ch"
        self.conv1 = ConvBnAct(in_ch=in_ch, out_ch=out_ch, k=3, s=1, p=1, act=True)
        self.conv2 = ConvBnAct(in_ch=out_ch, out_ch=out_ch, k=3, s=1, p=1, act=False)

    def forward(self, x):
        x_copy = x.clone()
        x = self.conv1(x)
        out = F.relu(self.conv2(x) + x_copy)
        return out


class ConvResBlock(nn.Module):
    """Residual Block with 1x1conv.
    Adds the 1x1 conv input to the output before applying the final act_fn.
    """

    def __init__(self, in_ch, out_ch, s=2):
        super(ConvResBlock, self).__init__()
        self.conv1 = ConvBnAct(in_ch=in_ch, out_ch=out_ch, k=3, s=s, p=1)
        self.conv2 = ConvBnAct(in_ch=out_ch, out_ch=out_ch, k=3, p=1, act=False)
        self.dsample = ConvBnAct(in_ch=in_ch, out_ch=out_ch, k=1, s=s, bn=False, act=False)

    def forward(self, x):
        x_copy = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        x_copy = self.dsample(x_copy)
        # print(x.shape, x_copy.shape)
        out = F.relu(x + x_copy)
        # print(out.shape)
        return out


def gem(x, p=3, eps=1e-6):
    """Generalized average pool: https://github.com/filipradenovic/cnnimageretrieval-pytorch"""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    """Generalized average pool: https://github.com/filipradenovic/cnnimageretrieval-pytorch"""

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


def resnet_block(in_ch, out_ch, n_blocks, first_block=False):
    """Helper function to customize the number of blocks in the resnet."""
    layers = []
    names = []
    for i in range(n_blocks):
        if i == 0 and not first_block:
            layers.append(ConvResBlock(in_ch, out_ch, s=2))
            names.append(f'conv_blk{i}')
        else:
            layers.append(BasicResBlock(out_ch, out_ch))
            names.append(f'reg_blk{i}')
    return list(zip(names, layers))


class ResNet18(nn.Module):
    def __init__(self, n_cls=2):
        super(ResNet18, self).__init__()
        params = dict(in_ch=3, out_ch=64, k=7, s=2, p=3)  # GoogLeNet

        self.l1 = nn.Sequential(ConvBnAct(**params), # x = [B, 3, 32, 32] # CHECK
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # -> [B, 512, 16, 16]
        self.l2 = nn.Sequential(OrderedDict(resnet_block(64, 64, 2, first_block=True)))  # -> [B, 64, 8, 8]
        self.l3 = nn.Sequential(OrderedDict(resnet_block(64, 128, 2)))  # -> [B, 128, 4, 4]
        self.l4 = nn.Sequential(OrderedDict(resnet_block(128, 256, 2)))  # -> [B, 256, 2, 2]
        self.l5 = nn.Sequential(OrderedDict(resnet_block(256, 512, 2)))  # -> [B, 512, 1, 1]
        self.pool = GeM()  # nn.AdaptiveAvgPool2d((1,1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512, n_cls)  # -> [512, n_cls]

    def forward(self, x):
        x = self.l5(self.l4(self.l3(self.l2(self.l1(x)))))
        return self.fc(self.flat(self.pool(x)))