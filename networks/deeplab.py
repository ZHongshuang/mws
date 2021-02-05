import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import torch.nn.functional as F
from .densenet import *
from .resnet import *
from .vgg import *
from .mobilenetv2 import *
from .tools import *
from .base_network import BaseNetwork

import numpy as np
import sys

thismodule = sys.modules[__name__]
import pdb


def proc_densenet(model):
    # dilation
    def remove_sequential(all_layers, network):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(all_layers, layer)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)

    model.features.transition2[-1].kernel_size = 1
    model.features.transition2[-1].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features.denseblock3)
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (2, 2)
            m.padding = (2, 2)

    model.features.transition3[-1].kernel_size = 1
    model.features.transition3[-1].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features.denseblock4)
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (4, 4)
            m.padding = (4, 4)
    model.classifier = None
    return model


def proc_vgg(model):
    # dilation
    def remove_sequential(all_layers, network):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(all_layers, layer)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)

    model.features[3][-1].kernel_size = 1
    model.features[3][-1].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features[4])
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (2, 2)
            m.padding = (2, 2)
    model.classifier = None
    return model


def proc_mobilenet2(model):
    # dilation
    def remove_sequential(all_layers, network):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(all_layers, layer)
            if isinstance(layer, InvertedResidual):
                remove_sequential(all_layers, layer.conv)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)

    model.features[7].conv[3].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features[8:14])
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (2, 2)
            m.padding = (2, 2)

    model.features[14].conv[3].stride = 1
    all_layers = []
    remove_sequential(all_layers, model.features[15:])
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            m.dilation = (4, 4)
            m.padding = (4, 4)
    model.classifier = None
    return model


procs = {
    'densenet169': proc_densenet,
    'vgg16': proc_vgg,
    'mobilenet2': proc_mobilenet2,
}



def batch_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"
    b_i, c, h, w = x.shape
    b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape
    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)
    out = F.conv2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding)
    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])
    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)
    return out

class DeepLab(nn.Module, BaseNetwork):
    def __init__(self, pretrained=True, c_output=21, base='densenet169'):
        super(DeepLab, self).__init__()
        dims = dim_dict[base][::-1]
        self.multi_preds = nn.ModuleList([nn.Conv2d(1664, c_output, kernel_size=3, dilation=dl, padding=dl)
                                    for dl in [6, 12, 18, 24]])
        self.upscale = nn.ConvTranspose2d(c_output, c_output, 16, 8, 4)
        self.apply(weight_init)
        self.feature = getattr(thismodule, base)(pretrained=pretrained)
        self.feature = procs[base](self.feature)
        self.apply(fraze_bn)

    def forward(self, x):
        x = self.feature(x)
        x = sum([f(x) for f in self.multi_preds])
        x = self.upscale(x)
        return x



if __name__ == "__main__":
    fcn = WSFCN2(base='densenet169').cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = fcn(Variable(x))
