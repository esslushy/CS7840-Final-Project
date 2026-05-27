import torch
import torch.nn as nn
from collections import OrderedDict


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, padding_mode="reflect")
        self.norm1 = nn.InstanceNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, padding_mode="reflect")
        self.norm2 = nn.InstanceNorm2d(dim)

    def forward(self, x, prefix=""):
        acts = OrderedDict()
        residual = x
        x = self.conv1(x)
        acts[f"{prefix}conv1"] = x.detach()
        x = self.norm1(x)
        acts[f"{prefix}norm1"] = x.detach()
        x = self.relu(x)
        acts[f"{prefix}relu"] = x.detach()
        x = self.conv2(x)
        acts[f"{prefix}conv2"] = x.detach()
        x = self.norm2(x)
        acts[f"{prefix}norm2"] = x.detach()
        x = x + residual
        acts[f"{prefix}residual"] = x.detach()
        return x, acts


class SmileToNeutralGenerator(nn.Module):
    def __init__(self, n_res_blocks=9):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 64, 7, padding=3, padding_mode="reflect")
        self.enc_norm1 = nn.InstanceNorm2d(64)
        self.enc_relu1 = nn.ReLU(inplace=True)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc_norm2 = nn.InstanceNorm2d(128)
        self.enc_relu2 = nn.ReLU(inplace=True)
        self.enc_conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.enc_norm3 = nn.InstanceNorm2d(256)
        self.enc_relu3 = nn.ReLU(inplace=True)

        # Transform
        self.res_blocks = nn.ModuleList([ResBlock(256) for _ in range(n_res_blocks)])

        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.dec_norm1 = nn.InstanceNorm2d(128)
        self.dec_relu1 = nn.ReLU(inplace=True)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_norm2 = nn.InstanceNorm2d(64)
        self.dec_relu2 = nn.ReLU(inplace=True)
        self.dec_conv3 = nn.Conv2d(64, 3, 7, padding=3, padding_mode="reflect")
        self.dec_tanh = nn.Tanh()

    def forward(self, x):
        acts = OrderedDict()

        # Encoder
        x = self.enc_conv1(x)
        acts["enc_conv1"] = x.detach()
        x = self.enc_norm1(x)
        acts["enc_norm1"] = x.detach()
        x = self.enc_relu1(x)
        acts["enc_relu1"] = x.detach()
        x = self.enc_conv2(x)
        acts["enc_conv2"] = x.detach()
        x = self.enc_norm2(x)
        acts["enc_norm2"] = x.detach()
        x = self.enc_relu2(x)
        acts["enc_relu2"] = x.detach()
        x = self.enc_conv3(x)
        acts["enc_conv3"] = x.detach()
        x = self.enc_norm3(x)
        acts["enc_norm3"] = x.detach()
        x = self.enc_relu3(x)
        acts["enc_relu3"] = x.detach()

        # Transform
        for i, block in enumerate(self.res_blocks):
            x, block_acts = block(x, prefix=f"res{i}.")
            acts.update(block_acts)

        # Decoder
        x = self.dec_conv1(x)
        acts["dec_conv1"] = x.detach()
        x = self.dec_norm1(x)
        acts["dec_norm1"] = x.detach()
        x = self.dec_relu1(x)
        acts["dec_relu1"] = x.detach()
        x = self.dec_conv2(x)
        acts["dec_conv2"] = x.detach()
        x = self.dec_norm2(x)
        acts["dec_norm2"] = x.detach()
        x = self.dec_relu2(x)
        acts["dec_relu2"] = x.detach()
        x = self.dec_conv3(x)
        acts["dec_conv3"] = x.detach()
        x = self.dec_tanh(x)
        acts["dec_tanh"] = x.detach()

        return x, acts
    

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, padding=1),
        )
 
    def forward(self, x):
        return self.net(x)
 
