# -*- encoding: utf-8 -*-
'''
-----------------------------------------------
* @File    :   reidnet_resnet.py
* @Time    :   2021/03/08 22:05:03
* @Author  :   zhang_jinlong
* @Version :   1.0
* @Contact :   zhangjinlongwin@163.com
* @Address ：  https://github.com/LiquorPerfect
-----------------------------------------------
'''

import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from torch.optim import lr_scheduler

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ReIDNetResNet(torchvision.models.ResNet):
    def __init__(self, block, layers, num_calsses=751, bottleneck=256):
        super.__init__(block, layers)
        self.num_calsses = num_calsses
        self.avgpool = nn.AdaptiveAvgPool1d((1, 1))

        self.tail = nn.Sequential(
            OrderedDict([("bottleneck", nn.Linear(self.inplanes, bottleneck)),
                         ("bn", nn.BatchNorm1d(self.inplanes, bottleneck)),
                         ("dropout", nn.Dropout(p=0.5))]))
        self.tfc = nn.Linear(bottleneck, self.num_calsses)

        for m in self.tail.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        for m in self.tfc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.001)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f = x

        x = self.tail(x)
        x = self.tfc(x)

        return x, f

    def feature_size(self):
        return 2048


def reidnet_resnet_50(model_path=None, num_calsses=751, **kwargs):
    m = ReIDNetResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3],
                      num_calsses, **kwargs)
    if model_path is None:
        m.load_state_dict(model_zoo.load_url(model_urls['resnet50']),
                          strict=False)
    else:
        m.load_state_dict(torch.load(model_path), strict=True)
    return m