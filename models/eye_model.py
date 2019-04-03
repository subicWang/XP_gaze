# -*- coding: utf-8 -*-
"""
Aouther: Subic
Time: 2019/3/2: 10:56
判别人眼的类型：0不可用， 1good, 2闭眼
"""
import torch.nn as nn
import math
import torch
import torch.nn as nn
from torchvision.models import resnet18


class EyeDistinguish(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True)
        self.bn0 = nn.BatchNorm2d(3)
        self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
                                  resnet.layer3)
        self.extra = nn.Sequential(nn.Linear(256*2*3, 3))
        
    def forward(self, x):
        x = self.bn0(x)
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.extra(x)

        return x

# class EyeDistinguish_dumped(nn.Module):
#     def __init__(self):
#         super(EyeDistinguish_dumped, self).__init__()
#         self.bn0 = nn.BatchNorm2d(3)
#
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(inplace=True)
#         # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#
#         self.conv2_1 = torch.nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#         self.conv2_2 = torch.nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv3_1 = torch.nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#         self.conv3_2 = torch.nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv4_1 = torch.nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True)
#         )
#         self.conv4_2 = torch.nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True)
#         )
#
#         self.avgpooling = nn.AvgPool2d((4, 6), stride=1)
#         self.fc = nn.Linear(256, 3)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         x = self.bn0(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.conv2_1(x)
#         x = self.conv2_2(x)
#         x = self.conv3_1(x)
#         x = self.conv3_2(x)
#
#         x = self.conv4_1(x)
#         x = self.conv4_2(x)
#
#         x = self.avgpooling(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x