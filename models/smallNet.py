"""
直接卷
"""
import torch.nn as nn
import math
import torch


class SmallNet1(nn.Module):
    def __init__(self):
        super(SmallNet1, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.extra1_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.extra1_bn1 = nn.BatchNorm2d(128)
        self.extra1_conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.extra1_bn2 = nn.BatchNorm2d(256)
        self.extra1_maxpool = nn.AvgPool2d((2, 3), stride=1)
        self.extra1_fc = nn.Linear(256, 1)
        self.extra1_sigmoid = nn.Sigmoid()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.avgpooling = nn.AvgPool2d((4, 6), stride=1)
        self.fc = nn.Linear(512, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool(x)

        y = self.extra1_conv1(x)
        y = self.extra1_bn1(y)
        y = self.relu(y)
        y = self.extra1_conv2(y)
        y = self.extra1_bn2(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = y.view(y.size(0), -1)
        y = self.extra1_fc(y)
        y = self.extra1_sigmoid(y)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, y


class SmallNet2(nn.Module):
    def __init__(self):
        super(SmallNet2, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2_1 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3_1 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv4_1 = torch.nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4_2 = torch.nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.extra1_conv1 = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.extra1_conv2 = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.extra1_maxpool = nn.AvgPool2d((2, 3), stride=1)
        self.extra1_fc = nn.Linear(128, 3)
        # self.extra1_sigmoid = nn.Sigmoid()
        #
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn4 = nn.BatchNorm2d(128)
        # self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn5 = nn.BatchNorm2d(256)
        # self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn6 = nn.BatchNorm2d(512)
        self.avgpooling = nn.AvgPool2d((4, 6), stride=1)
        self.fc = nn.Linear(256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        y = self.extra1_conv1(x)
        y = self.extra1_conv2(y)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        y = self.extra1_maxpool(y)
        y = y.view(y.size(0), -1)
        y = self.extra1_fc(y)
        # y = self.extra1_sigmoid(y)

        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, y


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = out + residual
        out = self.relu(out)
        return out


class SmallNet3(nn.Module):
    def __init__(self):
        super(SmallNet3, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)

        self.conv_base = torch.nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block1 = BasicBlock(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block2 = BasicBlock(128, 256)
        self.conv_label = torch.nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv_gaze = torch.nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.label_avgpool = nn.AvgPool2d((2, 3))
        self.label_fc = nn.Linear(256, 3)

        self.gaze_avgpool = nn.AvgPool2d((2, 3))
        self.gaze_fc = nn.Linear(256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv_base(x)

        x = self.maxpool1(x)
        x = self.block1(x)

        x = self.maxpool2(x)
        x = self.block2(x)

        gaze = self.conv_gaze(x)
        label = self.conv_label(x)

        label = self.label_avgpool(label)
        label = label.view(label.size(0), -1)
        label = self.label_fc(label)

        gaze = self.gaze_avgpool(gaze)
        gaze = gaze.view(gaze.size(0), -1)
        gaze = self.gaze_fc(gaze)

        return gaze, label


class SmallNet4OneEye(nn.Module):
    def __init__(self):
        super(SmallNet4OneEye, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)

        self.conv1_1 = torch.nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = torch.nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2_1 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3_1 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv4_1 = torch.nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4_2 = torch.nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.extra1_conv1 = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.extra1_conv2 = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.extra1_fc = nn.Linear(128*6, 2)
        self.fc = nn.Linear(256*6, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        y = self.extra1_conv1(x)
        y = self.extra1_conv2(y)
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        y = y.view(y.size(0), -1)
        y = self.extra1_fc(y)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, y


class SmallNet4TwoEyes(nn.Module):
    def __init__(self):
        super(SmallNet4TwoEyes, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)

        self.conv1_1 = torch.nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = torch.nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2_1 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3_1 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv4_1 = torch.nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4_2 = torch.nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.extra1_conv1 = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.extra1_conv2 = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.extra1_fc = nn.Linear(128*6*2, 6)
        self.fc = nn.Linear(256*6*2, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        y = self.extra1_conv1(x)
        y = self.extra1_conv2(y)
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        y = y.view(y.size(0), -1)
        y = self.extra1_fc(y)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, y




