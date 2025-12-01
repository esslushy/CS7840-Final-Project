import torch.nn as nn
import torch.nn.functional as F
import torch

class NaiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 10)

    def forward(self, x):
        layer_outs = []
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        out = nn.functional.softmax(x, dim=-1)
        layer_outs.append(out)
        return out, x, layer_outs
