import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange


def _gn_groups(channels, max_groups=8):
    """Pick the largest group count up to max_groups that divides channels."""
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# NaiveNet  (single linear layer)
# ---------------------------------------------------------------------------

class NaiveNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        acts = OrderedDict()
        out = self.conv1(x)
        acts["conv1"] = out.detach()
        return out, acts


# ---------------------------------------------------------------------------
# CNN  (conv stack with nonlinearities, no skip connections)
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self, width1=120, width2=84, in_channels=1, out_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width1, 3, padding=1)
        self.norm1 = nn.GroupNorm(_gn_groups(width1), width1)
        self.conv2 = nn.Conv2d(width1, width2, 3, padding=1)
        self.norm2 = nn.GroupNorm(_gn_groups(width2), width2)
        self.conv3 = nn.Conv2d(width2, width2, 3, padding=1)
        self.norm3 = nn.GroupNorm(_gn_groups(width2), width2)
        self.conv4 = nn.Conv2d(width2, width1, 3, padding=1)
        self.norm4 = nn.GroupNorm(_gn_groups(width1), width1)
        self.conv5 = nn.Conv2d(width1, width1 // 2, 3, padding=1)
        self.norm5 = nn.GroupNorm(_gn_groups(width1 // 2), width1 // 2)
        self.out_conv = nn.Conv2d(width1 // 2, out_channels, 1)

    def forward(self, x):
        acts = OrderedDict()
        x = self.conv1(x)
        acts["conv1"] = x.detach()
        x = self.norm1(x)
        acts["norm1"] = x.detach()
        x = F.silu(x)
        acts["silu1"] = x.detach()
        x = self.conv2(x)
        acts["conv2"] = x.detach()
        x = self.norm2(x)
        acts["norm2"] = x.detach()
        x = F.silu(x)
        acts["silu2"] = x.detach()
        x = self.conv3(x)
        acts["conv3"] = x.detach()
        x = self.norm3(x)
        acts["norm3"] = x.detach()
        x = F.silu(x)
        acts["silu3"] = x.detach()
        x = self.conv4(x)
        acts["conv4"] = x.detach()
        x = self.norm4(x)
        acts["norm4"] = x.detach()
        x = F.silu(x)
        acts["silu4"] = x.detach()
        x = self.conv5(x)
        acts["conv5"] = x.detach()
        x = self.norm5(x)
        acts["norm5"] = x.detach()
        x = F.silu(x)
        acts["silu5"] = x.detach()
        out = self.out_conv(x)
        return out, acts


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, width1=120, width2=84, in_channels=1, out_channels=2):
        super().__init__()
        # Encoder
        self.enc1_conv1 = nn.Conv2d(in_channels, width1, 3, padding=1)
        self.enc1_norm1 = nn.GroupNorm(_gn_groups(width1), width1)
        self.enc1_conv2 = nn.Conv2d(width1, width1, 3, padding=1)
        self.enc1_norm2 = nn.GroupNorm(_gn_groups(width1), width1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_conv1 = nn.Conv2d(width1, width2, 3, padding=1)
        self.enc2_norm1 = nn.GroupNorm(_gn_groups(width2), width2)
        self.enc2_conv2 = nn.Conv2d(width2, width2, 3, padding=1)
        self.enc2_norm2 = nn.GroupNorm(_gn_groups(width2), width2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.mid_conv1 = nn.Conv2d(width2, width2 * 2, 3, padding=1)
        self.mid_norm1 = nn.GroupNorm(_gn_groups(width2 * 2), width2 * 2)
        self.mid_conv2 = nn.Conv2d(width2 * 2, width2 * 2, 3, padding=1)
        self.mid_norm2 = nn.GroupNorm(_gn_groups(width2 * 2), width2 * 2)

        # Decoder
        self.up2 = nn.ConvTranspose2d(width2 * 2, width2, 2, stride=2)
        self.dec2_conv1 = nn.Conv2d(width2 * 2, width2, 3, padding=1)
        self.dec2_norm1 = nn.GroupNorm(_gn_groups(width2), width2)
        self.dec2_conv2 = nn.Conv2d(width2, width2, 3, padding=1)
        self.dec2_norm2 = nn.GroupNorm(_gn_groups(width2), width2)

        self.up1 = nn.ConvTranspose2d(width2, width1, 2, stride=2)
        self.dec1_conv1 = nn.Conv2d(width1 * 2, width1, 3, padding=1)
        self.dec1_norm1 = nn.GroupNorm(_gn_groups(width1), width1)
        self.dec1_conv2 = nn.Conv2d(width1, width1, 3, padding=1)
        self.dec1_norm2 = nn.GroupNorm(_gn_groups(width1), width1)

        self.out_conv = nn.Conv2d(width1, out_channels, 1)

    def forward(self, x):
        acts = OrderedDict()

        # Encoder 1
        x = self.enc1_conv1(x)
        acts["enc1_conv1"] = x.detach()
        x = self.enc1_norm1(x)
        acts["enc1_norm1"] = x.detach()
        x = F.silu(x)
        acts["enc1_silu1"] = x.detach()
        x = self.enc1_conv2(x)
        acts["enc1_conv2"] = x.detach()
        x = self.enc1_norm2(x)
        acts["enc1_norm2"] = x.detach()
        x = F.silu(x)
        acts["enc1_silu2"] = x.detach()
        skip1 = x
        x = self.pool1(x)

        # Encoder 2
        x = self.enc2_conv1(x)
        acts["enc2_conv1"] = x.detach()
        x = self.enc2_norm1(x)
        acts["enc2_norm1"] = x.detach()
        x = F.silu(x)
        acts["enc2_silu1"] = x.detach()
        x = self.enc2_conv2(x)
        acts["enc2_conv2"] = x.detach()
        x = self.enc2_norm2(x)
        acts["enc2_norm2"] = x.detach()
        x = F.silu(x)
        acts["enc2_silu2"] = x.detach()
        skip2 = x
        x = self.pool2(x)

        # Bottleneck
        x = self.mid_conv1(x)
        acts["mid_conv1"] = x.detach()
        x = self.mid_norm1(x)
        acts["mid_norm1"] = x.detach()
        x = F.silu(x)
        acts["mid_silu1"] = x.detach()
        x = self.mid_conv2(x)
        acts["mid_conv2"] = x.detach()
        x = self.mid_norm2(x)
        acts["mid_norm2"] = x.detach()
        x = F.silu(x)
        acts["mid_silu2"] = x.detach()

        # Decoder 2
        x = self.up2(x)
        if x.shape != skip2.shape:
            x = F.pad(x, [0, skip2.shape[3] - x.shape[3], 0, skip2.shape[2] - x.shape[2]])
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2_conv1(x)
        acts["dec2_conv1"] = x.detach()
        x = self.dec2_norm1(x)
        acts["dec2_norm1"] = x.detach()
        x = F.silu(x)
        acts["dec2_silu1"] = x.detach()
        x = self.dec2_conv2(x)
        acts["dec2_conv2"] = x.detach()
        x = self.dec2_norm2(x)
        acts["dec2_norm2"] = x.detach()
        x = F.silu(x)
        acts["dec2_silu2"] = x.detach()

        # Decoder 1
        x = self.up1(x)
        if x.shape != skip1.shape:
            x = F.pad(x, [0, skip1.shape[3] - x.shape[3], 0, skip1.shape[2] - x.shape[2]])
        x = torch.cat([x, skip1], dim=1)
        x = self.dec1_conv1(x)
        acts["dec1_conv1"] = x.detach()
        x = self.dec1_norm1(x)
        acts["dec1_norm1"] = x.detach()
        x = F.silu(x)
        acts["dec1_silu1"] = x.detach()
        x = self.dec1_conv2(x)
        acts["dec1_conv2"] = x.detach()
        x = self.dec1_norm2(x)
        acts["dec1_norm2"] = x.detach()
        x = F.silu(x)
        acts["dec1_silu2"] = x.detach()

        out = self.out_conv(x)
        return out, acts


# ---------------------------------------------------------------------------
# ViT helpers
# ---------------------------------------------------------------------------

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
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, prefix=""):
        acts = OrderedDict()
        x = self.norm(x)
        acts[f"{prefix}norm"] = x.detach()
        x = self.linear1(x)
        x = F.gelu(x)
        acts[f"{prefix}gelu"] = x.detach()
        x = self.linear2(x)
        acts[f"{prefix}linear"] = x.detach()
        return x, acts


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

    def forward(self, x, prefix=""):
        acts = OrderedDict()
        x = self.norm(x)
        acts[f"{prefix}norm"] = x.detach()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        acts[f"{prefix}attn"] = attn.detach()
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        acts[f"{prefix}out"] = out.detach()
        return out, acts


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


# ---------------------------------------------------------------------------
# ViT  (patch-based vision transformer → pixel-level output)
# ---------------------------------------------------------------------------

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, in_channels=1, out_channels=2, dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        patch_dim = in_channels * patch_height * patch_width
        self.out_channels = out_channels
        self.grid_h = image_height // patch_height
        self.grid_w = image_width // patch_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.image_height = image_height
        self.image_width = image_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=self.grid_h,
            w=self.grid_w,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.unpatch = nn.Linear(dim, out_channels * patch_height * patch_width)

    def forward(self, img):
        acts = OrderedDict()
        x = self.to_patch_embedding(img)
        acts["patch_embed"] = x.detach()
        x += self.pos_embedding.to(img.device, dtype=x.dtype)
        acts["pos_embed"] = x.detach()
        x, t_acts = self.transformer(x)
        acts.update(t_acts)
        x = self.unpatch(x)
        acts["unpatch"] = x.detach()

        # Reshape patches back to image
        x = x.view(-1, self.grid_h, self.grid_w, self.out_channels, self.patch_height, self.patch_width)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = x.view(-1, self.out_channels, self.image_height, self.image_width)

        return out, acts