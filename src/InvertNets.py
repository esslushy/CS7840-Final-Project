import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_features: int = 64):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
 
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
 
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
 
        self.layer4 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
 
        self.layer5 = nn.Sequential(
            nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_outs = []
        x = self.layer1(x)
        layer_outs.append(torch.flatten(x, 1))
        x = self.layer2(x)
        layer_outs.append(torch.flatten(x, 1))
        x = self.layer3(x)
        layer_outs.append(torch.flatten(x, 1))
        x = self.layer4(x)
        layer_outs.append(torch.flatten(x, 1))
        x = self.layer5(x)
        layer_outs.append(torch.flatten(x, 1))
        return x, layer_outs
