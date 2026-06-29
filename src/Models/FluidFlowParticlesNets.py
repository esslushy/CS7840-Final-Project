import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
from einops import rearrange


# ---------------------------------------------------------------------------
# NaiveNet  (single linear layer)
# ---------------------------------------------------------------------------

class NaiveNet(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)

    def forward(self, x):
        # x: (B, N, channels)
        acts = OrderedDict()
        out = self.fc1(x)
        acts["fc1"] = out.detach()
        return out, acts


# ---------------------------------------------------------------------------
# MLP  (shared MLP across points, no inter-point communication)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, width1=128, width2=64, channels=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, width1)
        self.norm1 = nn.LayerNorm(width1)
        self.fc2 = nn.Linear(width1, width2)
        self.norm2 = nn.LayerNorm(width2)
        self.fc3 = nn.Linear(width2, width2)
        self.norm3 = nn.LayerNorm(width2)
        self.fc4 = nn.Linear(width2, width1)
        self.norm4 = nn.LayerNorm(width1)
        self.fc5 = nn.Linear(width1, channels)

    def forward(self, x):
        # x: (B, N, channels)
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
        x = self.fc4(x)
        acts["fc4"] = x.detach()
        x = self.norm4(x)
        acts["norm4"] = x.detach()
        x = F.silu(x)
        acts["silu4"] = x.detach()
        out = self.fc5(x)
        return out, acts


# ---------------------------------------------------------------------------
# PointNet  (shared MLPs + global max pool + per-point prediction)
# ---------------------------------------------------------------------------

class PointNet(nn.Module):
    def __init__(self, width1=128, width2=64, channels=4):
        super().__init__()
        # Per-point encoder
        self.enc1 = nn.Linear(channels, width1)
        self.enc_norm1 = nn.LayerNorm(width1)
        self.enc2 = nn.Linear(width1, width2)
        self.enc_norm2 = nn.LayerNorm(width2)
        self.enc3 = nn.Linear(width2, width1)
        self.enc_norm3 = nn.LayerNorm(width1)

        # Per-point decoder (receives point features + global features)
        self.dec1 = nn.Linear(width1 + width1, width2)
        self.dec_norm1 = nn.LayerNorm(width2)
        self.dec2 = nn.Linear(width2, width2)
        self.dec_norm2 = nn.LayerNorm(width2)
        self.dec3 = nn.Linear(width2, channels)

    def forward(self, x):
        # x: (B, N, channels)
        acts = OrderedDict()

        # Encoder
        x = self.enc1(x)
        acts["enc1"] = x.detach()
        x = self.enc_norm1(x)
        acts["enc1_norm"] = x.detach()
        x = F.silu(x)
        acts["enc1_silu"] = x.detach()
        x = self.enc2(x)
        acts["enc2"] = x.detach()
        x = self.enc_norm2(x)
        acts["enc2_norm"] = x.detach()
        x = F.silu(x)
        acts["enc2_silu"] = x.detach()
        x = self.enc3(x)
        acts["enc3"] = x.detach()
        x = self.enc_norm3(x)
        acts["enc3_norm"] = x.detach()
        x = F.silu(x)
        acts["enc3_silu"] = x.detach()

        # Global max pool
        global_feat = x.max(dim=1, keepdim=True).values
        acts["global_pool"] = global_feat.detach()
        global_feat = global_feat.expand_as(x)

        # Concatenate per-point + global
        x = torch.cat([x, global_feat], dim=-1)
        acts["concat"] = x.detach()

        # Decoder
        x = self.dec1(x)
        acts["dec1"] = x.detach()
        x = self.dec_norm1(x)
        acts["dec1_norm"] = x.detach()
        x = F.silu(x)
        acts["dec1_silu"] = x.detach()
        x = self.dec2(x)
        acts["dec2"] = x.detach()
        x = self.dec_norm2(x)
        acts["dec2_norm"] = x.detach()
        x = F.silu(x)
        acts["dec2_silu"] = x.detach()
        out = self.dec3(x)
        return out, acts


# ---------------------------------------------------------------------------
# SetTransformer helpers
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
    def __init__(self, dim, heads=4, dim_head=32):
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


class TransformerBlock(nn.Module):
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
# SetTransformer
# ---------------------------------------------------------------------------

class SetTransformer(nn.Module):
    def __init__(self, dim=128, depth=1, heads=4, mlp_dim=128, dim_head=32, channels=4):
        super().__init__()
        self.input_proj = nn.Linear(channels, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.transformer = TransformerBlock(dim, depth, heads, dim_head, mlp_dim)
        self.output_proj = nn.Linear(dim, channels)

    def forward(self, x):
        # x: (B, N, channels)
        acts = OrderedDict()
        x = self.input_proj(x)
        acts["input_proj"] = x.detach()
        x = self.input_norm(x)
        acts["input_norm"] = x.detach()
        x, t_acts = self.transformer(x)
        acts.update(t_acts)
        out = self.output_proj(x)
        acts["output_proj"] = out.detach()
        return out, acts