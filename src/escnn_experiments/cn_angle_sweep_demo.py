"""
Equivariance error across the whole rotation group, measured by CKA alone.

TERMINOLOGY, because it matters here: LEE is Gruver et al.'s *Local Equivariance
Error*, defined as the Lie-derivative norm ||L_X f(x)|| / ||f(x)|| taken at the
identity. It is one scalar per model. The finite-angle quantity
||f(g.x) - rho(g) f(x)|| / ||rho(g) f(x)|| is a different thing, called
"finite-theta error" throughout this file and always qualified by which rho it
assumed. Both belong to the same rho-dependent family; neither is CKA.

Every other angle-resolved equivariance metric in this repo needs rho(g): the
finite-theta error compares f(g.x) against rho(g) f(x), so it can only be
evaluated where rho(g) is known -- at the group elements, and nowhere else. CKA needs no rho(g) at all. If
f(g.x) = rho(g) f(x) for ANY orthogonal rho(g), the centered Gram matrix of the
features is unchanged and CKA is exactly 1. That makes it the only metric here
that can be swept continuously over theta.

So sweep it. For C_N models with N = 1, 2, 4, 8, 16 (plus SO(2) as a floor),
walk theta from 0 to 360 in half-degree steps and plot 1 - CKA, normalized by the
SO(2) control's peak so that y = 1 is the measurement floor. That normalization is
what makes the curves comparable: raw, C1's peak is 417x the control's, so on one
linear axis C16 (29x) and the control are both flattened onto the baseline and
only C1/C2/C4 can be read. Dividing by the floor and going log keeps the
amplitude ordering -- 417x, 414x, 290x, 113x, 29x, 1x -- while making all six
legible. --norm none gives the raw linear version. The prediction:
C_N is exactly equivariant on its own N elements and nowhere else, so its curve
must touch zero at every multiple of 360/N and rise in between -- N lobes, each
of width 360/N, shrinking in amplitude as N grows and the model approaches the
SO(2) limit. rho(g) is never constructed; the group structure is recovered from
feature geometry alone.

Three things make this measurable rather than buried in resampling error.

1. ANNULAR-RING READOUT. Comparing features at theta requires no interpolation
   on the feature side: pool each hidden map over RINGS=4 concentric annuli.
   Each ring is rotation-invariant as a set, so exact equivariance gives
   Phi(g.x) = (rho_fiber(g) kron I_K) Phi(x), and rho_fiber of a regular_repr is
   a cyclic permutation -- orthogonal, hence CKA = 1 exactly. Measured floor is
   ~1e-4, 10x below a plain disk mean and 10x below un-rotating the feature map.

2. BAND-LIMITED SYNTHETIC DATA, ROTATED ANALYTICALLY. Each image is a sum of
   plane waves under a radial taper, so rotating it is just rotating the
   k-vectors -- exact, zero interpolation floor. (bicubic grid_sample is an
   acceptable fallback; bilinear costs C16 43% of its dip depth.)

3. LARGE KERNELS AT HIGH ANGULAR BANDWIDTH. This is the subtle one, and the
   direct analogue of the bandwidth confound in lee_bandwidth_confound_demo.py.
   With escnn's default frequencies_cutoff a KS=7 kernel carries angular
   frequencies only up to ~9, which is at or below C16's group Nyquist (N/2=8) --
   the C16 constraint becomes vacuous and the curve is a flat line at 1e-4. And
   with a small kernel, rotating the SAMPLED kernel by 360/N is badly resolved,
   so every model's dips land at multiples of 90 regardless of N (exactly the
   result of lie_vs_cka_group_aliasing_demo.py). KS=33 at FREQ_CUTOFF=24 on a
   129^2 grid fixes both.

Two findings worth reading off the plot:

C1's 180-degree notch is real, not a bug. C1 has no group, yet its error dips to
10% of peak at 180 (0.057 against a 0.570 peak) while sitting at 95% of peak at 90
and 270. Any rotation-invariant readout is dominated by the low
angular harmonics of the effective filter, and a 180-degree rotation maps
harmonic m to (-1)^m, preserving every even one. The notch survives all four
readouts tried (rings, disk-mean, un-rotation, angular-harmonic magnitude), so it
is a property of invariant readouts plus the square lattice, not of this metric.

At the dips, the model is NOT actually equivariant on the grid -- and CKA is
right anyway. Measured against escnn's own GeometricTensor.transform() (exact
rho(g)), via --rho-check:

                    err, exact rho    1-CKA
    C4  @90.0 deg         0.0000       0.0005     lattice rotation: both agree
    C8  @90.0             0.0000       0.0005     lattice rotation: both agree
    C8  @45.0             0.2346       0.0180     off-lattice group element
    C16 @22.5             0.3537       0.0198     off-lattice group element
    C16 @45.0             0.2384       0.0082     off-lattice group element

At the lattice rotations, where rho(g) is a pure index permutation, the two
metrics agree exactly at 0. Off the lattice they diverge, and the gap is the
cost of *interpolating* rho(g) onto a sampled feature map -- a cost the
rho-dependent metrics charge to the model. The ring readout never interpolates, so it does not pay it.

Run (escnn needs the sub-venv; the root venv has no lie_learn):
    src/escnn_experiments/.venv/bin/python3 src/escnn_experiments/cn_angle_sweep_demo.py
    ... --quick        smoke test, ~1 min
    ... --plot-only    re-plot from saved JSON without recomputing
    ... --norm none     raw 1-CKA on a linear axis (default divides by the SO(2)
                        control's peak and uses a log axis, which is the only way
                        C16 and the control are legible beside C1: they differ 417x)
    ... --lee          sweep the finite-theta error and LEE alongside CKA (2nd figure).
                       Raw error over theta, one ordinal colour ramp per metric.
                       Its stdout table scores the zeros: 1-CKA drops to 0.2-1.6%
                       of its own typical level at the group elements, while the
                       finite-theta error sits at 98-102% of its own at the same
                       angles -- it never registers one, not even the lattice
                       rotations where the exact rho reads 0. That table is only
                       as good as the angle grid; see lee_angles().
    ... --rho-check    finite-theta error under escnn's exact rho(g), at group elements
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from escnn import gspaces, nn as enn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import EquivarianceTracker

OUT_DIR = Path(__file__).resolve().parent / "out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- swept groups. -1 is SO(2), the floor control. -----------------------------
GROUPS = [1, 2, 4, 8, 16, -1]

# --- geometry / data. kmax must track FREQ_CUTOFF; at kmax=4 the C4 peak
#     collapses 40x. 24 cycles/image is 37% of Nyquist at S=129. ---------------
S = 129
KMAX = 24.0            # cycles across the image
N_WAVES = 64           # plane waves summed per image
TAPER = (0.55, 0.90)   # raised-cosine radial taper, in normalized radius
N_SAMPLES = 192        # n=512 gave no smoothness benefit over n=192
CHUNK = 48             # forward-pass chunk; C16 is 128ch x 129^2, ~11GB card

# --- model. NFIELDS is held fixed across N on purpose: matching total CHANNELS
#     instead (48//N) varies the number of independent base filters 48x and makes
#     the amplitude ordering NON-monotone (C4 would exceed C1). ----------------
NFIELDS = 8
KS = 33
FREQ_CUTOFF = 24.0
SO2_MAX_FREQ = 10
SO2_IRREPS = [0, 0, 1, 2, 3]   # 8 dims per field group x NFIELDS = 64 channels

# --- readout ------------------------------------------------------------------
RINGS = 4
R_MAX = 0.72

# --- sweep. 1-degree steps miss 22.5 / 67.5 / 112.5 ... entirely, so C16 never
#     samples half its own group and shows no dips at all. ---------------------
ANGLE_STEP = 0.5
N_SEEDS = 5            # seed SD is 15-30% of peak; 5-seed averaging cuts
                       # curve roughness from 0.031 to 0.0006


# ---------------------------------------------------------------------------
# Band-limited data, rotated analytically
# ---------------------------------------------------------------------------
def make_waves(n, seed, device):
    """Random plane-wave parameters: (freqs (n,M,2), phases (n,M), amps (n,M)).

    Radii are sampled in [0.15, 1] * KMAX/2 cycles per unit length (the image
    spans [-1,1], so width 2), keeping energy off DC without a hard low cut."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    f_max = KMAX / 2.0
    r = (0.15 + 0.85 * torch.rand(n, N_WAVES, generator=g)) * f_max
    a = torch.rand(n, N_WAVES, generator=g) * 2 * math.pi
    freqs = torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=-1)
    phases = torch.rand(n, N_WAVES, generator=g) * 2 * math.pi
    amps = torch.randn(n, N_WAVES, generator=g) / math.sqrt(N_WAVES)
    return freqs.to(device), phases.to(device), amps.to(device)


def coord_grid(size, device):
    c = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([xx, yy], dim=-1), torch.sqrt(xx ** 2 + yy ** 2)


def radial_taper(radius):
    r0, r1 = TAPER
    t = ((radius - r0) / (r1 - r0)).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * t))


def render(freqs, phases, amps, coords, window, theta):
    """Rotate the image by theta by rotating the k-vectors. Exact, no resampling.

    x_theta(p) = w(|p|) sum_j a_j cos(k_j . R_{-theta} p + phi_j)
               = w(|p|) sum_j a_j cos((R_theta k_j) . p + phi_j)
    """
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=freqs.dtype, device=freqs.device)
    fr = freqs @ R.T                                       # (n, M, 2)
    # phase(p) = 2*pi*(fx*x + fy*y) + phi
    ph = 2 * math.pi * torch.einsum("nmc,hwc->nmhw", fr, coords) + phases[..., None, None]
    img = (amps[..., None, None] * torch.cos(ph)).sum(1)   # (n, S, S)
    return (img * window)[:, None]


# ---------------------------------------------------------------------------
# Models: one R2Conv at high angular bandwidth, ReLU, a 1x1 R2Conv, ReLU.
# The nonlinearity is load-bearing -- it is what generates angular harmonics
# above N/2, i.e. what makes the group constraint bite. Without it C16 is flat.
# The second conv MUST be 1x1: a KS>=5 second layer destroys C16's 22.5 dip
# (and OOMs on an 11GB card anyway).
# ---------------------------------------------------------------------------
def build(N, seed, size, ks=KS, cutoff=FREQ_CUTOFF):
    torch.manual_seed(seed)
    if N == -1:
        gspace = gspaces.rot2dOnR2(N=-1, maximum_frequency=SO2_MAX_FREQ)
        hid = enn.FieldType(gspace, [gspace.irrep(k) for k in SO2_IRREPS] * NFIELDS)
        act = lambda t: enn.NormNonLinearity(t)
    else:
        gspace = gspaces.rot2dOnR2(N=N)
        hid = enn.FieldType(gspace, [gspace.regular_repr] * NFIELDS)
        act = lambda t: enn.ReLU(t)
    in_type = enn.FieldType(gspace, [gspace.trivial_repr])
    net = enn.SequentialModule(
        enn.R2Conv(in_type, hid, ks, padding=ks // 2, frequencies_cutoff=lambda r: cutoff),
        act(hid),
        enn.R2Conv(hid, hid, 1, frequencies_cutoff=lambda r: cutoff),
        act(hid),
    ).to(DEVICE)
    net.eval()
    return in_type, net


# ---------------------------------------------------------------------------
# Readout: mean over RINGS equal-area concentric annuli. Each ring is
# rotation-invariant as a set, so no interpolation is ever applied to features.
# ---------------------------------------------------------------------------
def ring_masks(radius):
    edges = R_MAX * torch.sqrt(torch.linspace(0.0, 1.0, RINGS + 1, device=radius.device))
    return torch.stack([(radius >= edges[k]) & (radius < edges[k + 1]) for k in range(RINGS)])


@torch.no_grad()
def ring_features(net, in_type, imgs, masks):
    """(n,1,S,S) -> (n, C*RINGS), chunked to stay inside GPU memory."""
    w = masks.float()
    w = w / w.sum((-2, -1), keepdim=True).clamp(min=1.0)   # (R, S, S)
    outs = []
    for i in range(0, imgs.shape[0], CHUNK):
        h = net(enn.GeometricTensor(imgs[i:i + CHUNK], in_type)).tensor  # (b, C, S, S)
        outs.append(torch.einsum("bchw,rhw->bcr", h, w).flatten(1))
    return torch.cat(outs)


def cka_error(X, Y):
    t = EquivarianceTracker(DEVICE)
    t.update(X, Y)
    if t.total_batches == 0:
        raise RuntimeError("tracker saw no batches")
    st = t.compute_stats()
    return 1.0 - st["linear_cka"], 1.0 - st["rbf_cka"]


# ---------------------------------------------------------------------------
def sweep(size, n, seeds, angles, ks, cutoff):
    coords, radius = coord_grid(size, DEVICE)
    window = radial_taper(radius)
    masks = ring_masks(radius)
    out = {str(N): {"linear": [], "rbf": []} for N in GROUPS}

    for si in range(seeds):
        freqs, phases, amps = make_waves(n, 1000 + si, DEVICE)
        x0 = render(freqs, phases, amps, coords, window, 0.0)
        for N in GROUPS:
            in_type, net = build(N, si, size, ks, cutoff)
            X = ring_features(net, in_type, x0, masks)
            lin, rbf = [], []
            for deg in angles:
                xt = render(freqs, phases, amps, coords, window, math.radians(deg))
                Y = ring_features(net, in_type, xt, masks)
                a, b = cka_error(X, Y)
                lin.append(a)
                rbf.append(b)
            out[str(N)]["linear"].append(lin)
            out[str(N)]["rbf"].append(rbf)
            del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print(f"  seed {si}  {'SO(2)' if N == -1 else f'C{N}':>6s}  "
                  f"peak={max(lin):.4f}  mean={float(np.mean(lin)):.4f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Plot. Series are ORDINAL (C1 < C2 < C4 < C8 < C16), so the correct encoding is
# a single-hue sequential ramp light->dark, not categorical hues. Steps are the
# blue 250/350/450/550/700 ramp; validated --ordinal (monotone lightness,
# adjacent dL >= 0.06, light end 2.06:1 on surface, hue spread 4 deg). Monotone
# lightness is also what makes it CVD-safe. SO(2) is a control, not a member of
# the family, so it wears muted ink and a dashed line.
# ---------------------------------------------------------------------------
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
INK, MUTED, GRID, SURFACE = "#26262b", "#898781", "#e1e0d9", "#fcfcfb"


def close_loop(y):
    """Append the theta=360 value by copying theta=0. The sweep covers [0, 360),
    so curves otherwise stop at 359.5 and the final lobe hangs open at the right
    edge; equivariance error is 360-periodic, so 360 is exactly 0's value."""
    return np.append(y, y[0])


def logsafe(y):
    """A log axis cannot draw an exact 0, and matplotlib's default is to break
    the line into a gap there -- which reads as a rendering fault rather than as
    the strongest result in the figure. Map zeros far below the axis instead, so
    each curve plunges off the bottom edge at its group elements; the rug panel
    underneath is what states that those dips are exactly 0."""
    return np.where(y > 0, y, 1e-12)


def plot(angles, data, stat, path, norm="floor"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 5.9in was sized for a title band and a four-line footnote; with those gone
    # the same panel height fits in 4.9, rather than stretching to fill it.
    fig = plt.figure(figsize=(11, 4.9), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1.0], height_ratios=[5.0, 1.25],
                          wspace=0.26, hspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    rug = fig.add_subplot(gs[1, 0], sharex=ax)
    axi = fig.add_subplot(gs[0, 1])
    for a in (ax, axi):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#c3c2b7")
        a.tick_params(colors=MUTED, labelcolor=INK, which="both")
    fig.patch.set_facecolor(SURFACE)

    n_seeds = len(data[str(GROUPS[0])][stat])
    finite = [N for N in GROUPS if N != -1]
    peaks, curves = {}, {}
    for N in GROUPS:
        y = close_loop(np.asarray(data[str(N)][stat]).mean(0))
        curves[N] = y
        peaks[N] = float(y.max())
    x = np.append(np.asarray(angles), 360.0)

    # Raw 1-CKA spans 417x from C1's peak to the SO(2) control's, so on one
    # linear axis C16 and the control are both pinned to the baseline and only
    # C1/C2/C4 are legible. Dividing by the control's peak puts every curve in
    # units of the measurement floor -- y=1 is "as equivariant as the exactly
    # equivariant model measures" -- and a log axis then fits all five at once
    # while KEEPING the amplitude-falls-with-N ordering that peak-normalizing
    # each curve to 1 would throw away.
    floor = peaks[-1]
    if norm == "floor":
        curves = {N: y / floor for N, y in curves.items()}
        peaks = {N: p / floor for N, p in peaks.items()}
        unit = "\u00d7 floor"
        fmt = lambda v: f"{v:,.0f}\u00d7"
    else:
        unit, fmt = "", lambda v: f"{v:.3f}"

    # SO(2) floor first, so the finite groups draw over it.
    sane = logsafe if norm == "floor" else (lambda y: y)
    ax.plot(x, sane(curves[-1]), color=MUTED, lw=1.6, ls="--", zorder=2,
            label=f"SO(2) control   peak {fmt(peaks[-1])}"
                  + ("  (defines the floor)" if norm == "floor" else ""))
    handles = []
    for i, N in reversed(list(enumerate(finite))):
        (h,) = ax.plot(x, sane(curves[N]), color=RAMP[i], lw=2.0,
                       zorder=3 + (len(finite) - i),
                       label=f"C{N}   peak {fmt(peaks[N])}   {N} zero{'s' if N > 1 else ''}")
        handles.append(h)
    handles.reverse()

    if norm == "floor":
        # One decade below the deepest resolved dip of any finite group; the
        # exact zeros run off the bottom edge, which the rug panel labels.
        lo = min(float(curves[N][curves[N] > 0].min()) for N in finite)
        bot = 10.0 ** math.floor(math.log10(lo))
        ax.set_yscale("log")
        ax.set_ylim(bot, max(peaks.values()) * 1.8)
        # Anything inside this band is indistinguishable from exact equivariance.
        ax.axhspan(bot, 1.0, color="#eceae1", zorder=1, lw=0)
        ax.text(357, bot * 1.18, "at or below the SO(2) floor", color=MUTED,
                fontsize=7.5, ha="right", va="bottom", zorder=8)
    else:
        ax.set_ylim(0, max(peaks.values()) * 1.42)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xticks(np.arange(0, 361, 22.5), minor=True)
    ax.grid(axis="x", which="major", color=GRID, lw=0.8, zorder=0)
    ax.grid(axis="x", which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("rotation angle  θ  (degrees)", color=INK)
    ax.set_ylabel((f"(1 − {stat} CKA) / SO(2) floor   (log)"
                   if norm == "floor" else
                   f"equivariance error   1 − {stat} CKA"), color=INK)
    # On the log axis the curves fill the panel and a 1.42x linear-style pad is
    # a sliver, so the legend moves out into the margin; the raw variant keeps
    # its original in-panel placement.
    where = ({"loc": "upper center"} if norm == "none" else
             {"loc": "lower center", "bbox_to_anchor": (0.5, 1.002)})
    ax.legend(handles=handles + [ax.lines[0]], ncol=3, frameon=False, fontsize=8,
              labelcolor=INK, columnspacing=1.4, handlelength=1.8, **where)
    ax.tick_params(labelbottom=False)
    ax.set_xlabel("")

    # The one anomaly worth calling out. Marked on the curve; prose goes in a
    # footnote rather than on top of the C4/C8 lobes.
    notch = int(np.argmin(np.abs(x - 180.0)))
    ax.plot([180], [curves[1][notch]], "o", ms=7, mfc="none", mec=RAMP[0],
            mew=1.8, zorder=10)

    for i, N in enumerate(finite):
        elems = [360.0 * k / N for k in range(N)] + [360.0]
        rug.vlines(elems, i + 0.18, i + 0.82, color=RAMP[i], lw=1.6)
        rug.text(-4, i + 0.5, f"C{N}", color=INK, fontsize=8, ha="right", va="center")
    rug.set_ylim(len(finite), 0)
    rug.set_yticks([])
    for side in ("top", "right", "left"):
        rug.spines[side].set_visible(False)
    rug.spines["bottom"].set_color("#c3c2b7")
    rug.set_facecolor(SURFACE)
    rug.tick_params(colors=MUTED, labelcolor=INK, which="both")
    rug.grid(axis="x", which="major", color=GRID, lw=0.8, zorder=0)
    rug.set_axisbelow(True)
    rug.set_xlabel("rotation angle  \u03b8  (degrees)", color=INK)

    # Inset: the headline claim, amplitude shrinking toward the SO(2) limit.
    nn = np.array(finite, dtype=float)
    axi.plot(nn, [peaks[N] for N in finite], "-o", color=RAMP[3], lw=1.8, ms=6,
             mfc=SURFACE, mew=1.8, zorder=3)
    axi.axhline(peaks[-1], color=MUTED, ls="--", lw=1.2, zorder=2)
    axi.text(16, peaks[-1], "SO(2) floor ", color=MUTED, fontsize=7.5,
             ha="right", va="bottom")
    axi.set_xscale("log", base=2)
    axi.set_xticks(nn)
    axi.set_xticklabels([f"C{N}" for N in finite], fontsize=8)
    axi.minorticks_off()
    if norm == "floor":
        axi.set_yscale("log")
        axi.set_ylim(0.55, max(peaks[N] for N in finite) * 2.6)
    else:
        axi.set_ylim(0, max(peaks[N] for N in finite) * 1.15)
    axi.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    axi.set_axisbelow(True)
    axi.set_title("peak error vs. group order" + (f"  ({unit})" if unit else ""),
                  fontsize=9, color=INK, pad=8)

    # No suptitle/subtitle to clear any more, so the axes take the space back;
    # the floor variant still leaves a strip above for its out-of-panel legend.
    fig.subplots_adjust(top=0.930 if norm == "none" else 0.900,
                        bottom=0.105, left=0.105, right=0.955)
    rb = rug.get_position()
    fig.text(0.012, (rb.y0 + rb.y1) / 2, "exact\nzeros", color=MUTED, fontsize=8,
             ha="left", va="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".png"), facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# The rho-dependent family this figure is a counterexample to.
#
# LEE (Local Equivariance Error) differentiates theta -> rho(theta)^-1
# f(rot_theta x) at theta = 0. That immediately exposes the structural problem:
# it is a derivative at the IDENTITY, so it returns one scalar per model with no
# theta argument at all. It cannot produce a curve over the group even in
# principle. And a Lie derivative needs a Lie algebra; C_N is discrete and has
# none, so what the derivative actually measures is deviation from SO(2), a
# group no C_N model belongs to.
#
# The finite-theta version has the complementary problem: it needs rho(theta),
# which for C_N regular fields simply does not exist off the N group elements.
# Gruver et al.'s implementation supplies the missing rho(theta) by assuming
# every channel is an independent scalar field, i.e. rho(theta) = spatial
# rotation. That IS evaluable at every theta -- and it is the assumption
# lie_vs_cka_escnn_exact_demo.py already shows to be wrong, because the correct
# rho for a regular field also permutes the fiber.
#
# So the family gets three shots and misses three ways: blind (LEE is one
# scalar), undefined (no rho off-group), or wrong (naive rho). --lee measures
# all three.
#
# The figure plots both raw, on one linear axis. Normalizing each metric by its
# own scale was tried twice (per model, then per metric pooled) and read worse
# both times: it buys a level comparison the two do not support anyway -- the
# finite-theta error is a norm ratio saturating near 1, 1-CKA a similarity
# deficit at 1e-2 -- at the cost of the panel's directly readable amplitudes.
# The zeros comparison it was meant to serve lives in the stdout table instead.
#
# Both variants below rotate the FEATURE map by grid_sample. That interpolation
# is unavoidable for any rho-dependent metric, and is precisely what the ring
# readout is built to avoid.
# ---------------------------------------------------------------------------
LEE_N, LEE_SEEDS, LEE_STEP, LEE_CHUNK, LIE_EPS = 48, 3, 2.0, 16, 0.02
LEE_OFFSETS = (0.5, 1.5, 4.0)   # sampled either side of every group element


def lee_angles():
    """The LEE_STEP grid UNION every C_N group element, plus offsets around each.

    A uniform grid will not do for this figure. At LEE_STEP=2 the only sampled
    multiples of 22.5 are 0/90/180/270, so C8 lands on 4 of its 8 elements and
    C16 on 4 of 16 -- the same aliasing ANGLE_STEP warns about for the main
    sweep, and fatal here, because the claim IS that 1-CKA vanishes exactly on
    the elements. Every element of every swept group is a multiple of 360/16, so
    putting all 16 on the grid covers C1, C2, C4, C8 and C16 at once. The
    offsets resolve how narrow each dip is; anything off the base grid is
    excluded from theta-averages, which assume uniform spacing."""
    a = set(np.arange(0.0, 360.0, LEE_STEP).tolist())
    for k in range(16):
        e = 360.0 * k / 16.0
        a.add(e)
        for d in LEE_OFFSETS:
            a.add((e + d) % 360.0)
            a.add((e - d) % 360.0)
    return sorted(a)


def base_mask(a):
    """Which angles lie on the uniform LEE_STEP grid. lee_angles() clusters extra
    samples around the group elements, so a plain mean or median over the full
    grid is weighted toward the dips and is not a "typical level" of anything."""
    return np.isclose(np.mod(np.asarray(a), LEE_STEP), 0.0)


def elem_indices(a, N):
    """(grid indices of the sampled C_N elements, angles of the unsampled ones).

    theta=0 is excluded throughout: every metric is trivially 0 at the identity,
    so counting it would flatter both equally and prove nothing.

    The missing list is returned rather than swallowed because a uniform
    LEE_STEP=2 grid contains no multiple of 22.5 except 0/90/180/270, so C8 lands
    on 4 of its 8 elements and C16 on 4 of 16 -- and the 4 it lands on are the
    lattice rotations, the easy ones. Nearest-sample substitution is not an
    option: the dips are about 2 degrees wide (C4 reads 0.0005 at 90 but 0.020 at
    88), so scoring an element 1 degree away would report a true zero as a miss.
    Callers draw the gap; lee_angles() removes it on a rebuild."""
    a = np.asarray(a)
    idx, missing = [], []
    for k in range(1, N):
        e = 360.0 * k / N
        i = int(np.argmin(np.abs(a - e)))
        (idx.append(i) if abs(a[i] - e) < 1e-6 else missing.append(e))
    return np.array(idx, dtype=int), missing


def rotate_feat(t, theta):
    """Spatial rotation of a feature map, bicubic. Bilinear costs C16 43% of its
    dip depth, so it is not an acceptable substitute here."""
    th = torch.as_tensor(theta, dtype=t.dtype, device=t.device)
    zero = torch.zeros((), dtype=t.dtype, device=t.device)
    m = torch.stack([torch.stack([torch.cos(th), torch.sin(th), zero]),
                     torch.stack([-torch.sin(th), torch.cos(th), zero])])[None]
    m = m.expand(t.shape[0], -1, -1)
    grid = F.affine_grid(m, list(t.size()), align_corners=True)
    return F.grid_sample(t, grid, align_corners=True, mode="bicubic")


@torch.no_grad()
def finite_angle_error(net, in_type, y0, xt, theta, disk):
    """Relative equivariance error at a FINITE angle: ||f(rot.x) - rot.f(x)|| /
    ||rot.f(x)||, with rho(theta) assumed to be spatial rotation alone.

    This is NOT LEE. LEE (Local Equivariance Error) is defined as the
    Lie-derivative norm ||L_X f||/||f|| -- see lie_derivative() below. This is
    the finite-angle analogue, which is what you get if you try to extend LEE's
    assumed rho(theta) away from the identity. Restricted to the same disk the
    rings cover, so both metrics are compared on identical support."""
    num = den = 0.0
    for i in range(0, xt.shape[0], LEE_CHUNK):
        lhs = net(enn.GeometricTensor(xt[i:i + LEE_CHUNK], in_type)).tensor
        rhs = rotate_feat(y0[i:i + LEE_CHUNK], theta)
        num += float((((lhs - rhs) * disk) ** 2).sum())
        den += float(((rhs * disk) ** 2).sum())
    return math.sqrt(num / max(den, 1e-30))


@torch.no_grad()
def lie_derivative(net, in_type, waves, coords, window, disk):
    """LEE proper: Local Equivariance Error = ||L_X f(x)|| / ||f(x)||, the
    relative norm of the Lie derivative at the identity. One scalar per model. (torch's jvp raises NotImplementedError on
    grid_sampler_2d, which is why the repo uses finite differences throughout.)"""
    def f(th):
        x = render(*waves, coords, window, th)
        outs = []
        for i in range(0, x.shape[0], LEE_CHUNK):
            outs.append(rotate_feat(net(enn.GeometricTensor(x[i:i + LEE_CHUNK],
                                                            in_type)).tensor, -th))
        return torch.cat(outs)
    d = (f(LIE_EPS) - f(-LIE_EPS)) / (2 * LIE_EPS)
    return float(((d * disk) ** 2).sum().sqrt() / ((f(0.0) * disk) ** 2).sum().sqrt())


def lee_sweep(angles):
    """CKA and finite-angle error on identical data, plus LEE (the Lie-derivative
    norm at the identity). JSON key "lee" is kept for the Lie-derivative series'
    sibling for backward compatibility with already-written runs."""
    coords, radius = coord_grid(S, DEVICE)
    window, masks = radial_taper(radius), ring_masks(radius)
    disk = (radius <= R_MAX).float()
    out = {str(N): {"cka": [], "lee": [], "lie": []} for N in GROUPS}
    for si in range(LEE_SEEDS):
        waves = make_waves(LEE_N, 1000 + si, DEVICE)
        x0 = render(*waves, coords, window, 0.0)
        for N in GROUPS:
            in_type, net = build(N, si, S)
            y0 = torch.cat([net(enn.GeometricTensor(x0[i:i + LEE_CHUNK], in_type)).tensor
                            for i in range(0, LEE_N, LEE_CHUNK)])
            X = ring_features(net, in_type, x0, masks)
            cka, lee = [], []
            for deg in angles:
                th = math.radians(deg)
                xt = render(*waves, coords, window, th)
                cka.append(cka_error(X, ring_features(net, in_type, xt, masks))[0])
                lee.append(finite_angle_error(net, in_type, y0, xt, th, disk))
            lie = lie_derivative(net, in_type, waves, coords, window, disk)
            out[str(N)]["cka"].append(cka)
            out[str(N)]["lee"].append(lee)
            out[str(N)]["lie"].append(lie)
            del net, y0
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print(f"  seed {si}  {'SO(2)' if N == -1 else f'C{N}':>6s}  "
                  f"CKA peak={max(cka):.4f}  fin-err mean={float(np.mean(lee)):.4f}  "
                  f"Lie={lie:.4f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Cross-check: LEE under escnn's own exact rho(g), at the same group elements
# where the CKA sweep reads ~0. Needs the rho(g) the sweep never touches, which
# is the point -- applying it to a sampled feature map costs an interpolation
# that LEE then charges to the model.
#
# NOTE ON ORIENTATION: the analytic renderer and escnn's element indexing turn
# opposite ways, so rho(g) must be built from elements[(-k) % N], not
# elements[k]. Getting this backwards reads ~1.05 everywhere except 180 degrees
# (which is its own inverse, so it silently passes). C4 at 90 is the anchor: it
# is a lattice rotation and MUST be exactly 0.
# ---------------------------------------------------------------------------
@torch.no_grad()
def rho_check(n=32, seeds=2):
    coords, radius = coord_grid(S, DEVICE)
    window, masks = radial_taper(radius), ring_masks(radius)
    print(f"{'model':>6s} {'theta':>7s} {'err, exact rho':>15s} {'1-CKA, rings':>13s}"
          f" {'(wrong orient.)':>16s}")
    for N, ks in [(4, [1]), (8, [1, 2]), (16, [1, 2])]:
        acc = {}
        for si in range(seeds):
            f, ph, a = make_waves(n, 1000 + si, DEVICE)
            x0 = render(f, ph, a, coords, window, 0.0)
            in_type, net = build(N, si, S)
            hid = net.out_type
            elems = list(in_type.gspace.fibergroup.elements)
            y0 = net(enn.GeometricTensor(x0, in_type)).tensor
            X = ring_features(net, in_type, x0, masks)
            for k in ks:
                deg = 360.0 * k / N
                xt = render(f, ph, a, coords, window, math.radians(deg))
                lhs = net(enn.GeometricTensor(xt, in_type)).tensor
                vals = []
                for el in (elems[(-k) % N], elems[k]):
                    rhs = enn.GeometricTensor(y0, hid).transform(el).tensor
                    vals.append(((lhs - rhs).norm() / rhs.norm()).item())
                cka = cka_error(X, ring_features(net, in_type, xt, masks))[0]
                acc.setdefault(deg, []).append([vals[0], cka, vals[1]])
            del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        for deg, v in sorted(acc.items()):
            v = np.array(v).mean(0)
            print(f"{'C' + str(N):>6s} {deg:7.2f} {v[0]:15.4f} {v[1]:13.4f} {v[2]:16.4f}",
                  flush=True)
    print("\n  C4@90 and C8@90 are lattice rotations and read exactly 0 under both metrics.")
    print("  Off the lattice the two diverge: the exact-rho error charges the model for")
    print("  the cost of interpolating rho(g) onto a sampled grid, which rings never pay.")


# Categorical slots 1-3 (blue / orange / aqua), the only three that clear the
# all-pairs CVD and normal-vision floors in both modes. Metric identity is also
# carried by line style, so it never rests on hue alone.
CAT = {"cka": "#2a78d6", "lee": "#eb6834", "lie": "#1baf7a"}


def elem_ratios(a, y, N, base):
    """(value at each non-identity C_N element / typical level, typical level,
    elements the grid misses).

    "Typical level" is the median over the uniform subgrid, which makes the two
    metrics -- 1-CKA at 1e-2 and a finite-theta error at ~1.0 -- comparable as
    fractions of their own scale. Returned per element rather than reduced,
    because the spread is the finding: 1-CKA is a true zero at the lattice
    rotations and only a shallow dip at the off-lattice elements, where a C_N
    model on a square grid genuinely is not equivariant (--rho-check measures
    0.354 there against escnn's exact rho). A single worst-case number would
    report that honest shallowness as a failure to locate the element."""
    med = float(np.median(y[base]))
    idx, missing = elem_indices(a, N)
    return y[idx] / med, med, missing


# One ordinal ramp per metric instead of one hue at three alphas. Alpha was the
# problem: 0.50/0.75/1.00 of the same hue is a ~0.1 lightness spread against the
# surface, and the C4 and C8 curves were not separable where they overlap. These
# are real steps -- each ramp validated --ordinal (monotone lightness, adjacent
# dL >= 0.06, light end >= 2:1 on the surface, hue spread <= 6 deg), and every
# blue-vs-orange pair clears CVD dE 16-27 against a >= 8 target. Order is
# C4 -> C8 -> C16 -> SO(2), which is also the group-order progression, so the
# ramp direction carries meaning rather than just separating lines.
LEE_RAMP = {
    "cka": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    "lee": ["#ef8853", "#e2621f", "#b0410f", "#7a2c08"],
}


def plot_lee(angles, data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.append(np.asarray(angles), 360.0)
    finite = [N for N in GROUPS if N != -1]
    shown = [4, 8, 16]
    fig = plt.figure(figsize=(11, 4.7), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.35], wspace=0.24)
    ax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for a in (ax, axb):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#c3c2b7")
        a.tick_params(colors=MUTED, labelcolor=INK)
        a.grid(color=GRID, lw=0.8, zorder=0)
        a.set_axisbelow(True)
    fig.patch.set_facecolor(SURFACE)

    def series(N, k):
        return close_loop(np.asarray(data[str(N)][k]).mean(0))

    for i, N in enumerate(shown):
        ax.plot(x, series(N, "lee"), color=LEE_RAMP["lee"][i], lw=1.7, ls="--",
                zorder=3, label=f"finite-\u03b8 err, naive \u03c1   C{N}")
        ax.plot(x, series(N, "cka"), color=LEE_RAMP["cka"][i], lw=2.0,
                zorder=4, label=f"1 \u2212 CKA   C{N}")
    # The SO(2) control, on both metrics. It is the sharpest line in the panel:
    # its CKA is pinned at ~0 for every theta (equivariant everywhere, which is
    # the truth), while its finite-theta error rides ABOVE every finite group --
    # it ranks the only exactly-equivariant model as the least equivariant one.
    # It takes the dark end of each ramp, since SO(2) is where C_N is heading.
    dot = (0, (1.4, 1.4))
    ax.plot(x, series(-1, "lee"), color=LEE_RAMP["lee"][3], lw=2.4, ls=dot,
            zorder=6, label="finite-\u03b8 err, naive \u03c1   SO(2)")
    ax.plot(x, series(-1, "cka"), color=LEE_RAMP["cka"][3], lw=2.4, ls=dot,
            zorder=7, label="1 \u2212 CKA   SO(2)")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, max(series(N, "lee").max() for N in shown + [-1]) * 1.52)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel("rotation angle  \u03b8  (degrees)", color=INK)
    ax.set_ylabel("relative equivariance error", color=INK)
    ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=7,
              labelcolor=INK, columnspacing=1.0, handlelength=2.4)

    order = finite + [-1]
    nn = np.arange(len(order))
    base = base_mask(angles)
    for label, key, vals, mk in [
        ("1 \u2212 CKA (mean over \u03b8)", "cka",
         [float(np.asarray(data[str(N)]["cka"]).mean(0)[base].mean()) for N in order], "-o"),
        ("finite-\u03b8 err, naive \u03c1 (mean)", "lee",
         [float(np.asarray(data[str(N)]["lee"]).mean(0)[base].mean()) for N in order], "-s"),
        ("LEE = \u2016L$_X$f\u2016/\u2016f\u2016  (per rad)", "lie",
         [float(np.mean(data[str(N)]["lie"])) for N in order], "-^"),
    ]:
        # The summary panel is one mark per metric, so it keeps the categorical
        # slots; only the theta panel needed the ramps.
        axb.plot(nn, vals, mk, color=CAT[key], lw=1.8, ms=6, mfc=SURFACE, mew=1.8,
                 zorder=3, label=label)
    axb.set_xticks(nn)
    axb.set_xticklabels([f"C{N}" for N in finite] + ["SO(2)"], fontsize=8)
    axb.axvline(len(finite) - 0.5, color=GRID, lw=1.0, zorder=1)
    axb.set_yscale("log")
    axb.set_ylabel("equivariance error  (log)", color=INK, fontsize=8)
    axb.legend(loc="lower left", frameon=False, fontsize=7, labelcolor=INK,
               handlelength=2.0)

    fig.subplots_adjust(top=0.965, bottom=0.115, left=0.075, right=0.985)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".png"), facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="smoke test (~1 min); C16 is under-resolved at 65^2, see note")
    p.add_argument("--plot-only", action="store_true", help="re-plot from saved JSON")
    p.add_argument("--stat", default="linear", choices=["linear", "rbf"])
    p.add_argument("--norm", default="floor", choices=["floor", "none"],
                   help="floor (default): plot 1-CKA in units of the SO(2) "
                        "control's peak on a log axis, so all six curves are "
                        "legible at once; none: raw 1-CKA on a linear axis")
    p.add_argument("--rho-check", action="store_true",
                   help="LEE under escnn's exact rho(g) at the group elements")
    p.add_argument("--lee", action="store_true",
                   help="sweep LEE (naive rho) and the Lie derivative alongside CKA")
    args = p.parse_args()

    if args.rho_check:
        rho_check()
        return

    if args.lee:
        jp = OUT_DIR / "cn_angle_sweep_lee.json"
        if args.plot_only:
            blob = json.loads(jp.read_text())
            ang, ld = blob["angles"], blob["data"]
        else:
            ang = lee_angles()
            print(f"device={DEVICE}  n={LEE_N}  seeds={LEE_SEEDS}  angles={len(ang)}")
            ld = lee_sweep(ang)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            jp.write_text(json.dumps({"angles": ang, "data": ld}))
            print(f"  wrote {jp}")
        plot_lee(ang, ld, OUT_DIR / "cn_angle_sweep_lee.pdf")
        # Means over the uniform subgrid only; the extra samples lee_angles()
        # clusters at the group elements would pull a full-grid mean down.
        bm = base_mask(ang)
        print(f"\n{'model':>7s} {'mean 1-CKA':>11s} {'mean fin-err':>13s} {'LEE':>8s}"
              f"   {'worst elem, as % of own median':>32s}")
        for N in GROUPS:
            nm = "SO(2)" if N == -1 else f"C{N}"
            cka = np.asarray(ld[str(N)]["cka"]).mean(0)
            fin = np.asarray(ld[str(N)]["lee"]).mean(0)
            if N == -1 or N == 1:
                # SO(2) is equivariant at every theta and C1 only at the identity;
                # neither has a non-identity element for a metric to find.
                tail = "   (no non-identity elements to find)"
            else:
                r_c, _, miss = elem_ratios(ang, cka, N, bm)
                r_f = elem_ratios(ang, fin, N, bm)[0]
                tail = (f"   1-CKA {100 * r_c.min():5.1f}-{100 * r_c.max():<5.1f}"
                        f"  fin-err {100 * r_f.min():5.1f}-{100 * r_f.max():<5.1f}"
                        + (f"  ({len(r_c)}/{N - 1} elems sampled)" if miss else ""))
            print(f"{nm:>7s} {cka[bm].mean():11.4f} {fin[bm].mean():13.4f} "
                  f"{np.mean(ld[str(N)]['lie']):8.4f}{tail}")
        return

    if args.quick:   # ks/cutoff must scale with the grid, not stay at KS=33
        size, n, seeds, step, ks, cutoff = 65, 64, 1, 2.5, 17, 16.0
    else:
        size, n, seeds, step, ks, cutoff = S, N_SAMPLES, N_SEEDS, ANGLE_STEP, KS, FREQ_CUTOFF
    tag = "cn_angle_sweep" + ("_quick" if args.quick else "")
    json_path = OUT_DIR / f"{tag}.json"
    angles = list(np.arange(0.0, 360.0, step))

    if args.plot_only:
        blob = json.loads(json_path.read_text())
        angles, data = blob["angles"], blob["data"]
    else:
        print(f"device={DEVICE}  grid={size}^2  n={n}  seeds={seeds}  "
              f"angles={len(angles)} @ {step} deg")
        data = sweep(size, n, seeds, angles, ks, cutoff)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({
            "angles": angles, "data": data,
            "config": {"size": size, "n": n, "seeds": seeds, "step": step,
                       "kmax": KMAX, "ks": ks, "freq_cutoff": cutoff,
                       "nfields": NFIELDS, "rings": RINGS, "r_max": R_MAX},
        }))
        print(f"  wrote {json_path}")

    suffix = "_raw" if args.norm == "none" else ""
    plot(angles, data, args.stat, OUT_DIR / f"{tag}_{args.stat}{suffix}.pdf", args.norm)
    if args.quick:
        print("\n  NOTE: --quick runs a 65^2 grid, which is under-resolved for C16 "
              "(expect\n  contrast near 1x there). It checks that the pipeline runs, "
              "not the result.\n  C1-C8 should still show exact zeros at their group elements.")

    # Contrast report: how deep is each group's own dip relative to its lobes?
    x = np.asarray(angles)
    floor = float(np.asarray(data["-1"][args.stat]).mean(0).max())
    print(f"\n{'model':>7s} {'peak':>9s} {'peak/floor':>11s} {'worst elem':>11s} "
          f"{'min lobe':>9s} {'contrast':>9s}")
    for N in GROUPS:
        y = np.asarray(data[str(N)][args.stat]).mean(0)
        if N == -1:
            print(f"{'SO(2)':>7s} {y.max():9.5f} {1.0:11.1f} {'--':>11s} {'--':>9s}"
                  f"   (defines floor)")
            continue
        elem = np.array([abs((x - 360.0 * k / N + 180) % 360 - 180).argmin() for k in range(N)])
        worst = y[elem].max()
        # lowest local maximum between consecutive group elements
        lobes = [y[(x > 360.0 * k / N) & (x < 360.0 * (k + 1) / N)] for k in range(N)]
        min_lobe = min(float(l.max()) for l in lobes if l.size)
        c = "exact 0" if worst < 1e-9 else f"{min_lobe / worst:.1f}x"
        print(f"{'C' + str(N):>7s} {y.max():9.5f} {y.max() / floor:11.1f} {worst:11.5f} "
              f"{min_lobe:9.5f} {c:>9s}")


if __name__ == "__main__":
    main()
