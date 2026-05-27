import torch
import torch.nn as nn
from collections import OrderedDict


class CNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_features: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        acts = OrderedDict()
        x = self.conv1(x)
        acts["conv1"] = x.detach()
        x = self.relu1(x)
        acts["relu1"] = x.detach()
        x = self.conv2(x)
        acts["conv2"] = x.detach()
        x = self.relu2(x)
        acts["relu2"] = x.detach()
        x = self.conv3(x)
        acts["conv3"] = x.detach()
        x = self.bn3(x)
        acts["bn3"] = x.detach()
        x = self.relu3(x)
        acts["relu3"] = x.detach()
        x = self.conv4(x)
        acts["conv4"] = x.detach()
        x = self.bn4(x)
        acts["bn4"] = x.detach()
        x = self.relu4(x)
        acts["relu4"] = x.detach()
        x = self.conv5(x)
        acts["conv5"] = x.detach()
        x = self.sigmoid(x)
        acts["sigmoid"] = x.detach()
        return x, acts