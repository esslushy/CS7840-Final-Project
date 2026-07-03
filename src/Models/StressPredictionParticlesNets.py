import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
from einops import rearrange


# ---------------------------------------------------------------------------
# NaiveNet  (single per-point linear layer)
# ---------------------------------------------------------------------------

class NaiveNet(nn.Module):
    """Per-point linear map. No interaction between particles."""

    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x: (B, N, in_channels)
        acts = OrderedDict()
        out = self.linear1(x)
        acts["linear1"] = out.detach()
        return out, acts


# ---------------------------------------------------------------------------
# MLP  (shared per-point MLP, still no cross-particle interaction)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Point-wise MLP applied identically to every particle."""

    def __init__(self, width1=128, width2=64, in_channels=4, out_channels=3):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, width1)
        self.norm1 = nn.LayerNorm(width1)
        self.fc2 = nn.Linear(width1, width2)
        self.norm2 = nn.LayerNorm(width2)
        self.fc3 = nn.Linear(width2, width2)
        self.norm3 = nn.LayerNorm(width2)
        self.out = nn.Linear(width2, out_channels)

    def forward(self, x):
        # x: (B, N, in_channels)
        acts = OrderedDict()
        x = self.fc1(x)
        acts["fc1"] = x.detach()
        x = self.norm1(x)
        acts["norm1"] = x.detach()
        x = F.silu(x)
        acts["silu1"] = x.detach()
        x = self.fc2(x)
        acts["fc2"] = x.detach()
        x = self.norm2(x)
        acts["norm2"] = x.detach()
        x = F.silu(x)
        acts["silu2"] = x.detach()
        x = self.fc3(x)
        acts["fc3"] = x.detach()
        x = self.norm3(x)
        acts["norm3"] = x.detach()
        x = F.silu(x)
        acts["silu3"] = x.detach()
        out = self.out(x)
        return out, acts


# ---------------------------------------------------------------------------
# PointNet  (per-point encoder + global max-pool context + per-point decoder)
# ---------------------------------------------------------------------------

class PointNet(nn.Module):
    """
    Segmentation-style PointNet: a shared per-point encoder produces local
    features, a permutation-invariant global feature is pooled across particles
    (max over the set) and concatenated back to every point, and a per-point
    decoder produces the output. The global pooling gives each particle access
    to set-level context, which is needed to invert the force -> stress
    (divergence) relationship.
    """

    def __init__(self, width1=128, width2=64, in_channels=4, out_channels=3):
        super().__init__()
        # per-point encoder
        self.enc1 = nn.Linear(in_channels, width1)
        self.enc_norm1 = nn.LayerNorm(width1)
        self.enc2 = nn.Linear(width1, width2)
        self.enc_norm2 = nn.LayerNorm(width2)
        # per-point decoder over [local ; global]
        self.dec1 = nn.Linear(width2 * 2, width1)
        self.dec_norm1 = nn.LayerNorm(width1)
        self.dec2 = nn.Linear(width1, width1)
        self.dec_norm2 = nn.LayerNorm(width1)
        self.out = nn.Linear(width1, out_channels)

    def forward(self, x):
        # x: (B, N, in_channels)
        acts = OrderedDict()

        # encoder
        h = self.enc1(x)
        acts["enc1"] = h.detach()
        h = self.enc_norm1(h)
        acts["enc_norm1"] = h.detach()
        h = F.silu(h)
        acts["enc_silu1"] = h.detach()
        h = self.enc2(h)
        acts["enc2"] = h.detach()
        h = self.enc_norm2(h)
        acts["enc_norm2"] = h.detach()
        local = F.silu(h)
        acts["enc_silu2"] = local.detach()

        # permutation-invariant global feature (max over particles)
        glob = local.max(dim=1, keepdim=True).values      # (B, 1, width2)
        acts["global"] = glob.detach()
        glob_exp = glob.expand(-1, local.shape[1], -1)
        h = torch.cat([local, glob_exp], dim=-1)           # (B, N, 2*width2)
        acts["concat"] = h.detach()

        # decoder
        h = self.dec1(h)
        acts["dec1"] = h.detach()
        h = self.dec_norm1(h)
        acts["dec_norm1"] = h.detach()
        h = F.silu(h)
        acts["dec_silu1"] = h.detach()
        h = self.dec2(h)
        acts["dec2"] = h.detach()
        h = self.dec_norm2(h)
        acts["dec_norm2"] = h.detach()
        h = F.silu(h)
        acts["dec_silu2"] = h.detach()

        out = self.out(h)
        return out, acts


# ---------------------------------------------------------------------------
# Set transformer helpers  (operate on (B, N, dim) token sequences)
# ---------------------------------------------------------------------------

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
# SetTransformer  (self-attention across particles -> per-point output)
# ---------------------------------------------------------------------------

class SetTransformer(nn.Module):
    """
    Permutation-equivariant transformer over the particle set. No positional
    embedding is used: each particle already carries its (x, y) position in the
    input channels, and the set has no intrinsic ordering. Self-attention lets
    every particle attend to the whole set (global context), and a per-point
    head maps the final tokens to the stress output.
    """

    def __init__(self, dim, depth, heads, mlp_dim, in_channels=4, out_channels=3, dim_head=64):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.LayerNorm(dim),
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.out = nn.Linear(dim, out_channels)

    def forward(self, x):
        # x: (B, N, in_channels)
        acts = OrderedDict()
        x = self.embed(x)
        acts["embed"] = x.detach()
        x, t_acts = self.transformer(x)
        acts.update(t_acts)
        out = self.out(x)
        acts["out_proj"] = out.detach()
        return out, acts