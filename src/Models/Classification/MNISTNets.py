import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange


class CNN(nn.Module):
    def __init__(self, width1=120, width2=84):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, width1)
        self.fc2 = nn.Linear(width1, width2)
        self.fc3 = nn.Linear(width2, 10)

    def forward(self, x):
        acts = OrderedDict()
        x = self.conv1(x)
        acts["conv1"] = x.detach()
        x = F.relu(x)
        acts["relu1"] = x.detach()
        x = self.pool(x)
        acts["pool1"] = x.detach()
        x = self.conv2(x)
        acts["conv2"] = x.detach()
        x = F.relu(x)
        acts["relu2"] = x.detach()
        x = self.pool(x)
        acts["pool2"] = x.detach()
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        acts["fc1"] = x.detach()
        x = F.relu(x)
        acts["relu3"] = x.detach()
        x = self.fc2(x)
        acts["fc2"] = x.detach()
        x = F.relu(x)
        acts["relu4"] = x.detach()
        logits = self.fc3(x)
        acts["fc3"] = logits.detach()
        out = F.softmax(logits, dim=-1)
        acts["softmax"] = out.detach()
        return out, logits, acts


class NaiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        acts = OrderedDict()
        x = torch.flatten(x, 1)
        logits = self.fc1(x)
        acts["fc1"] = logits.detach()
        out = F.softmax(logits, dim=-1)
        acts["softmax"] = out.detach()
        return out, logits, acts


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        acts = OrderedDict()
        for i, (attn, ff) in enumerate(self.layers):
            attn_out, attn_acts = attn(x, prefix=f"block{i}.attn.")
            acts.update(attn_acts)
            x = attn_out + x
            acts[f"block{i}.attn.residual"] = x.detach()
            ff_out, ff_acts = ff(x, prefix=f"block{i}.ff.")
            acts.update(ff_acts)
            x = ff_out + x
            acts[f"block{i}.ff.residual"] = x.detach()
        x = self.norm(x)
        acts["final_norm"] = x.detach()
        return x, acts


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        acts = OrderedDict()
        x = self.to_patch_embedding(img)
        acts["patch_embed"] = x.detach()
        x += self.pos_embedding.to(img.device, dtype=x.dtype)
        acts["pos_embed"] = x.detach()
        x, t_acts = self.transformer(x)
        acts.update(t_acts)
        x = x.mean(dim=1)
        acts["mean_pool"] = x.detach()
        logits = self.linear_head(x)
        acts["linear_head"] = logits.detach()
        out = F.softmax(logits, dim=-1)
        acts["softmax"] = out.detach()
        return out, logits, acts