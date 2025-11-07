import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        layer_outs = []
        x = self.pool(F.relu(self.conv1(x)))
        layer_outs.append(torch.flatten(x, 1))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        layer_outs.append(x)
        x = F.relu(self.fc1(x))
        layer_outs.append(x)
        x = F.relu(self.fc2(x))
        layer_outs.append(x)
        x = self.fc3(x)
        out = torch.nn.functional.softmax(x, dim=-1)
        layer_outs.append(out)
        return out, x, layer_outs
