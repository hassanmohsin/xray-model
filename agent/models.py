from collections import OrderedDict
from math import floor

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def resnet18(pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    if pretrained:
        for name, param in model.named_parameters():
            if 'bn' not in name:  # DON'T freeze BN layers
                param.requires_grad = False

    model.fc = nn.Sequential(OrderedDict([
        ('dropout1', nn.Dropout(0.5)),
        ('fc1', nn.Linear(512, 256)),
        ('activation1', nn.ReLU()),
        ('dropout2', nn.Dropout(0.3)),
        ('fc2', nn.Linear(256, 128)),
        ('activation2', nn.ReLU()),
        ('fc3', nn.Linear(128, 1))
    ]))

    return model


def resnet34(pretrained=False):
    model = models.resnet34(pretrained=pretrained)
    if pretrained:
        for name, param in model.named_parameters():
            if 'bn' not in name:  # DON'T freeze BN layers
                param.requires_grad = False

    model.fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(512, 256)),
        ('activation1', nn.ReLU()),
        ('dropout2', nn.Dropout(0.7)),
        ('fc2', nn.Linear(256, 128)),
        ('activation2', nn.ReLU()),
        ('fc3', nn.Linear(128, 1))
    ]))

    return model


def resnet50(pretrained=False):
    model = models.resnet50(pretrained=pretrained)
    if pretrained:
        for name, param in model.named_parameters():
            if 'bn' not in name:  # DON'T freeze BN layers
                param.requires_grad = False

    model.fc = nn.Sequential(
        OrderedDict(
            [
                ('dropout1', nn.Dropout(0.5)),
                ('fc1', nn.Linear(2048, 1024)),
                ('activation1', nn.ReLU()),
                ('dropout2', nn.Dropout(0.3)),
                ('fc2', nn.Linear(1024, 256)),
                ('activation2', nn.ReLU()),
                ('dropout3', nn.Dropout(0.3)),
                ('fc3', nn.Linear(256, 128)),
                ('activation3', nn.ReLU()),
                ('fc4', nn.Linear(128, 1))
            ]
        )
    )

    return model


def resnet101(pretrained):
    model = models.resnet101(pretrained=pretrained)
    if pretrained:
        for name, param in model.named_parameters():
            if 'bn' not in name:  # DON'T freeze BN layers
                param.requires_grad = False

    model.fc = nn.Sequential(
        OrderedDict(
            [
                ('dropout1', nn.Dropout(0.5)),
                ('fc1', nn.Linear(2048, 1024)),
                ('activation1', nn.ReLU()),
                ('dropout2', nn.Dropout(0.3)),
                ('fc2', nn.Linear(1024, 256)),
                ('activation2', nn.ReLU()),
                ('dropout3', nn.Dropout(0.3)),
                ('fc3', nn.Linear(256, 128)),
                ('activation3', nn.ReLU()),
                ('fc4', nn.Linear(128, 1))
            ]
        )
    )

    return model


def resnet152(pretrained=False):
    model = models.resnet152(pretrained=pretrained)
    if pretrained:
        for name, param in model.named_parameters():
            if 'bn' not in name:  # DON'T freeze BN layers
                param.requires_grad = False

    model.fc = nn.Sequential(
        OrderedDict(
            [
                ('dropout1', nn.Dropout(0.5)),
                ('fc1', nn.Linear(2048, 1024)),
                ('activation1', nn.ReLU()),
                ('dropout2', nn.Dropout(0.3)),
                ('fc2', nn.Linear(1024, 256)),
                ('activation2', nn.ReLU()),
                ('dropout3', nn.Dropout(0.3)),
                ('fc3', nn.Linear(256, 128)),
                ('activation3', nn.ReLU()),
                ('fc4', nn.Linear(128, 1))
            ]
        )
    )

    return model


def wide_resnet101_2(pretrained=False):
    model = models.wide_resnet101_2(pretrained=pretrained)
    if pretrained:
        for name, param in model.named_parameters():
            if 'bn' not in name:  # DON'T freeze BN layers
                param.requires_grad = False

    model.fc = nn.Sequential(
        OrderedDict(
            [
                ('dropout1', nn.Dropout(0.5)),
                ('fc1', nn.Linear(2048, 1024)),
                ('activation1', nn.ReLU()),
                ('dropout2', nn.Dropout(0.3)),
                ('fc2', nn.Linear(1024, 256)),
                ('activation2', nn.ReLU()),
                ('dropout3', nn.Dropout(0.3)),
                ('fc3', nn.Linear(256, 128)),
                ('activation3', nn.ReLU()),
                ('fc4', nn.Linear(128, 1))
            ]
        )
    )

    return model


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, 2)  # 247
        self.pool1 = nn.MaxPool2d(2, 2)  # 123
        self.conv2 = nn.Conv2d(8, 16, 3, 1)  # 120
        self.pool2 = nn.MaxPool2d(2, 2)  # 60
        self.conv3 = nn.Conv2d(16, 32, 3, 1)  # 57
        self.pool3 = nn.MaxPool2d(2, 2)  # 28
        self.fc1 = nn.Linear(32 * 29 * 29, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModelOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, 2)  # 125
        self.pool1 = nn.MaxPool2d(2, 2)  # 62
        self.conv2 = nn.Conv2d(8, 16, 3, 1)  # 59
        self.pool2 = nn.MaxPool2d(2, 2)  # 29
        self.conv3 = nn.Conv2d(16, 32, 3, 1)  # 26
        self.pool3 = nn.MaxPool2d(2, 2)  # 13
        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModelTwo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, 2)  # 247
        self.pool1 = nn.MaxPool2d(2, 2)  # 123
        self.conv2 = nn.Conv2d(8, 16, 3, 1)  # 120
        self.pool2 = nn.MaxPool2d(2, 2)  # 60
        self.conv3 = nn.Conv2d(16, 32, 3, 1)  # 57
        self.pool3 = nn.MaxPool2d(2, 2)  # 28
        self.fc1 = nn.Linear(32 * 29 * 29, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.30)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model(model_name='', pretrained=False):
    if model_name == "baseline":
        model = BaselineModel()
        model.apply(initialize_weights)
        return model
    elif model_name == "resnet18":
        return resnet18(pretrained)
    elif model_name == "resnet34":
        return resnet34(pretrained)
    elif model_name == "resnet50":
        return resnet50(pretrained)
    elif model_name == "resnet101":
        return resnet101(pretrained)
    elif model_name == "resnet152":
        return resnet152(pretrained)
    elif model_name == "wide_resnet101_2":
        return wide_resnet101_2(pretrained)
    else:
        raise NotImplementedError("Selected model has not been implemented yet!")
