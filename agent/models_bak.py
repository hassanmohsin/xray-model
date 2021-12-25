import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['BaselineModel', 'ModelOne', 'ModelTwo']


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, 2)  # 254
        self.pool1 = nn.MaxPool2d(2, 2)  # 127
        self.conv2 = nn.Conv2d(8, 16, 3, 1)  # 125
        self.pool2 = nn.MaxPool2d(2, 2)  # 62
        self.conv3 = nn.Conv2d(16, 32, 3, 1)  # 60
        self.pool3 = nn.MaxPool2d(2, 2)  # 30
        self.fc1 = nn.Linear(32 * 30 * 30, 64)
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
        self.conv1 = nn.Conv2d(3, 8, 5, 2)  # 128
        self.pool1 = nn.MaxPool2d(2, 2)  # 64
        self.conv2 = nn.Conv2d(8, 16, 3, 1)  # 31
        self.pool2 = nn.MaxPool2d(2, 2)  # 15
        self.conv3 = nn.Conv2d(16, 32, 3, 1)  # 13
        self.pool3 = nn.MaxPool2d(2, 2)  # 6
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
        self.conv1 = nn.Conv2d(3, 8, 5, 2)  # 254
        self.pool1 = nn.MaxPool2d(2, 2)  # 127
        self.conv2 = nn.Conv2d(8, 16, 3, 1)  # 125
        self.pool2 = nn.MaxPool2d(2, 2)  # 62
        self.conv3 = nn.Conv2d(16, 32, 3, 1)  # 60
        self.pool3 = nn.MaxPool2d(2, 2)  # 30
        self.fc1 = nn.Linear(32 * 30 * 30, 64)
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
