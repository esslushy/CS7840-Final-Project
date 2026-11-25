"""
Does CKA measure invariance to a NON-geometric transform? Test case: fonts.

Every CKA number in the sweep pairs x with g.x for a rotation g. Here the
transform is "render the same digit in a different font". That is not a group
action, so there is no rho(g) and family-A metrics (LEE / Lie derivative) cannot
even be written down. CKA needs no rho, so it can be applied. The open question
is whether it TRACKS font invariance or merely reports class clustering.

Setup (everything is rendered here, nothing is downloaded):

  fonts     System TTF/OTF/TTC files, filtered to ones whose ten digit glyphs
            are non-blank, pairwise distinct and recognisable against the
            cross-font median glyph (drops Wingdings, tofu, pictogram fonts).
            Italic/oblique styles are dropped, since a shear is geometric.
            Near-duplicate files are dropped. The rest are split BY FAMILY into
            train fonts and held-out fonts, so "Arial Bold" cannot leak into the
            held-out set while "Arial" trains.
  images    28x28, white on black, glyph bbox scaled to 18-22 px and shifted
            +-2 px (MNIST-style). Training images are always jittered.
  models    Models/Classification/MNISTNets.CNN (LeNet), three of them:
              untrained    random init
              one_font     trained on ONE train font (chosen per seed)
              many_fonts   trained on ALL train fonts, same image budget
            Held-out-font accuracy is the ground truth for "is it font
            invariant?". CKA should agree with it.
  eval      N pairs from held-out fonts only. Row i is
              X_i = render(digit d_i, font a_i, jitter j_i)
              Y_i = render(digit d_i, font b_i, jitter j_i),   a_i != b_i
            The jitter is SHARED within a pair, so the font is the only thing
            that differs between X_i and Y_i.

Pairings scored at every layer (linear and RBF CKA, via the sweep's own
unbiased-HSIC + pooled-median-sigma rule from utils.EquivarianceTracker,
full batch so the value is deterministic):

  font       (X_i, Y_i)               the measurement
  cls-shuf   (X_i, Y_pi(i)), pi a derangement WITHIN each digit class:
             same class, no per-sample correspondence. This is the crux null.
  delta      font - cls-shuf: correspondence CKA sees beyond class identity
  shuf       (X_i, Y_pi(i)), pi a global derangement: the CKA floor
  rot90      (X_i, rot90(X_i)): geometric reference on the same model

plus a local, rank-based second opinion that is not a Gram-alignment score:

  NN all     fraction of i whose Euclidean nearest Y is Y_i    (chance 1/N)
  NN cls     same, searching only the Y of the same digit    (chance ~10/N)

Run (from anywhere, root venv):
    python font_cka_demo.py
    python font_cka_demo.py --quick          # smoke test, ~1 minute
    python font_cka_demo.py --font-dir /usr/share/fonts --seeds 5

Results (Windows 11 system fonts: 334 files -> 177 usable, 123 train / 54
held-out; 3 seeds, N = 2000, 20000 training images x 5 epochs; linear CKA at
relu4, the deepest hidden layer; mean over seeds):

                  held-out   font    cls-shuf   delta    NN cls
  eval jitter ON (default)
    untrained      0.1045   0.3949   0.1665   +0.2283   0.0523
    one_font       0.7147   0.5414   0.4481   +0.0934   0.0210
    many_fonts     0.9755   0.7670   0.7084   +0.0586   0.0265
  eval jitter OFF (--no-jitter)
    untrained      0.0998   0.4526   0.4676   -0.0151   0.0002
    one_font       0.7857   0.6039   0.6231   -0.0192   0.0000
    many_fonts     0.9740   0.7984   0.8065   -0.0081   0.0003

Reading:
  1. Raw font-pair CKA ranks the models correctly, but only deep in the
     network. many_fonts > one_font > untrained matches held-out-font accuracy
     at fc2, relu4, fc3 and softmax, under both kernels, with and without
     jitter. At fc1 and relu3 it MISRANKS: one_font scores below the untrained
     net (linear, jitter on: 0.3026 vs 0.4082 at fc1). So CKA can be applied to
     a non-geometric transform, and at the layers near the readout it gives the
     right ranking.
  2. The pairing contributes nothing. The class-matched shuffle, where Y
     is a different, uncorresponding sample of the same digit, scores almost the
     same (0.7670 vs 0.7084). With jitter off it scores slightly HIGHER (0.8065
     vs 0.7984). At softmax the two agree to 1e-3 (0.9254 vs 0.9241).
  3. The whole positive delta is the shared jitter, not the font. It is largest
     for the UNTRAINED net, which keeps the size/position that X_i and Y_i
     share. It shrinks as training discards that nuisance. With the jitter
     removed it goes to ~0 (slightly negative) for every model. NN retrieval
     tells the same story: above chance only through the jitter, and at or below
     chance without it.
  Taken together: for a font change, CKA measures CROSS-FONT CLASS ALIGNMENT,
  meaning how far "same digit, any font" collapses to the same Gram structure.
  That is a legitimate class-level notion of font invariance, and it tracks
  accuracy. But it is not the instance-level x <-> T(x) correspondence that the
  rotation CKA is read as. Two fonts of one digit share nothing but the
  class, so there is no instance-level correspondence left to measure. The
  floor is also high: an untrained net already scores 0.39-0.45, so the
  number is only meaningful relative to that baseline.
"""
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
from utils import EquivarianceTracker, set_seed, unbiased_hsic  # noqa: E402
from Models.Classification.MNISTNets import CNN  # noqa: E402

GLYPH_PT, CANVAS = 96, 192
IMG = 28
BLOCKLIST = ("wingding", "webding", "symbol", "marlett", "mdl2", "fluent", "emoji",
             "specialty", "bookshelf", "mt extra", "dingbat", "ornament", "braille",
             "barcode", "music", "math")
SLANTED = ("italic", "oblique", "slanted")
SUMMARY_LAYERS = ("relu4", "fc3")
MODELS = ("untrained", "one_font", "many_fonts")


# --------------------------------------------------------------------------- fonts

def raw_glyphs(path):
    """High-res crops of the ten digit glyphs, or None if the font is unusable."""
    try:
        font = ImageFont.truetype(path, GLYPH_PT)
        family, style = font.getname()
    except (OSError, ValueError):
        return None
    crops = []
    for d in range(10):
        img = Image.new("L", (CANVAS, CANVAS), 0)
        try:
            ImageDraw.Draw(img).text((CANVAS // 2, CANVAS // 2), str(d), fill=255,
                                     font=font, anchor="mm")
        except (OSError, ValueError):
            return None
        bbox = img.getbbox()
        if bbox is None or min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < 4 and d != 1:
            return None
        crops.append(img.crop(bbox))
    return family or Path(path).stem, style or "", crops


def place(crop, size, dx, dy):
    """Scale the glyph's longer side to `size` px and paste it centred + (dx, dy)."""
    w, h = crop.size
    s = size / max(w, h)
    g = crop.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    canvas = Image.new("L", (IMG, IMG), 0)
    canvas.paste(g, ((IMG - g.width) // 2 + dx, (IMG - g.height) // 2 + dy))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def discover_fonts(font_dir, min_recognised):
    from matplotlib import font_manager
    paths = sorted(set(font_manager.findSystemFonts(fontpaths=font_dir, fontext="ttf")))
    dropped = defaultdict(int)
    cands = []
    for p in paths:
        r = raw_glyphs(p)
        if r is None:
            dropped["unloadable/blank"] += 1
            continue
        family, style, crops = r
        name = f"{family} {style}".lower()
        if any(b in name for b in BLOCKLIST):
            dropped["blocklisted name"] += 1
            continue
        if any(s in name for s in SLANTED):
            dropped["italic/oblique"] += 1
            continue
        canon = np.stack([place(c, 20, 0, 0).ravel() for c in crops])  # (10, 784)
        cands.append(dict(path=p, family=family, style=style, crops=crops, canon=canon))

    # recognisable: each glyph's nearest cross-font median prototype is its own digit
    protos = np.median(np.stack([c["canon"] for c in cands]), axis=0)  # (10, 784)
    kept = []
    for c in cands:
        dists = ((c["canon"][:, None, :] - protos[None]) ** 2).sum(-1)
        own = np.sort(((c["canon"][:, None] - c["canon"][None]) ** 2).sum(-1)
                      + np.eye(10) * 1e9, axis=1)[:, 0]
        if own.min() < 1.0:  # two digits render (almost) identically -> tofu-like
            dropped["glyphs not distinct"] += 1
        elif (dists.argmin(1) == np.arange(10)).sum() < min_recognised:
            dropped["digits unrecognisable"] += 1
        elif any(np.abs(c["canon"] - k["canon"]).mean() < 0.01 for k in kept):
            dropped["duplicate of a kept font"] += 1
        else:
            kept.append(c)
    return kept, len(paths), dict(dropped)


def split_by_family(fonts, eval_frac, seed):
    fams = sorted({f["family"] for f in fonts})
    rng = np.random.default_rng(seed)
    rng.shuffle(fams)
    eval_fams, n_eval = set(), 0
    for fam in fams:
        if n_eval >= eval_frac * len(fonts):
            break
        eval_fams.add(fam)
        n_eval += sum(f["family"] == fam for f in fonts)
    train = [i for i, f in enumerate(fonts) if f["family"] not in eval_fams]
    held = [i for i, f in enumerate(fonts) if f["family"] in eval_fams]
    return train, held


# --------------------------------------------------------------------------- data

def jitter(rng, n, on=True):
    if not on:
        return np.full(n, 20), np.zeros(n, int), np.zeros(n, int)
    return rng.integers(18, 23, n), rng.integers(-2, 3, n), rng.integers(-2, 3, n)


def render(fonts, font_ids, digits, sizes, dxs, dys):
    out = np.stack([place(fonts[f]["crops"][d], s, x, y)
                    for f, d, s, x, y in zip(font_ids, digits, sizes, dxs, dys)])
    return torch.from_numpy(out)[:, None]


def random_images(fonts, font_pool, n, rng):
    ids = rng.choice(font_pool, n)
    digits = rng.integers(0, 10, n)
    return render(fonts, ids, digits, *jitter(rng, n)), torch.from_numpy(digits)


def eval_pairs(fonts, held, n, rng, use_jitter):
    digits = rng.integers(0, 10, n)
    a = rng.integers(0, len(held), n)
    b = (a + rng.integers(1, len(held), n)) % len(held)  # b != a
    a, b = np.asarray(held)[a], np.asarray(held)[b]
    j = jitter(rng, n, use_jitter)
    return render(fonts, a, digits, *j), render(fonts, b, digits, *j), digits


# --------------------------------------------------------------------------- models

def train(fonts, pool, args, rng, device):
    net = CNN().to(device)
    if pool is None:
        return net
    X, y = random_images(fonts, pool, args.train_images, rng)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    for _ in range(args.epochs):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), 128):
            idx = perm[s:s + 128]
            _, logits, _ = net(X[idx].to(device))
            loss = F.cross_entropy(logits, y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


@torch.no_grad()
def accuracy(net, X, y, device):
    net.eval()
    hits = sum((net(X[s:s + 500].to(device))[1].argmax(1).cpu() == y[s:s + 500]).sum().item()
               for s in range(0, len(X), 500))
    return hits / len(X)


@torch.no_grad()
def features(net, X, device):
    net.eval()
    out = defaultdict(list)
    for s in range(0, len(X), 500):
        for k, v in net(X[s:s + 500].to(device))[2].items():
            out[k].append(v.flatten(1).double().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


# --------------------------------------------------------------------------- metrics

def pair_grams(X, Y):
    """Everything EquivarianceTracker.update computes, kept so permuted pairings
    can reuse it. The pooled median sigma is permutation-invariant, so it is the
    same for font / cls-shuf / shuf, exactly as the tracker would compute it."""
    n = len(X)
    P = torch.cat([X, Y])
    D = torch.cdist(P, P)
    sigma = D[torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)].median().item()
    sigma = sigma or 1e-6
    rbf = lambda D2: torch.exp(-D2 / (2 * sigma ** 2))
    return dict(K_lin=X @ X.T, L_lin=Y @ Y.T, K_rbf=rbf(D[:n, :n] ** 2),
                L_rbf=rbf(D[n:, n:] ** 2), DXY=D[:n, n:], sigma=sigma)


def cka(g, perm=None):
    L_lin, L_rbf = g["L_lin"], g["L_rbf"]
    if perm is not None:
        L_lin, L_rbf = L_lin[perm][:, perm], L_rbf[perm][:, perm]
    K_lin, K_rbf = g["K_lin"], g["K_rbf"]
    h = lambda A, B: unbiased_hsic(A, B).item()
    t = EquivarianceTracker("cpu")
    t.accumulate(h(K_lin, L_lin), h(K_lin, K_lin), h(L_lin, L_lin),
                 h(K_rbf, L_rbf), h(K_rbf, K_rbf), h(L_rbf, L_rbf), g["sigma"])
    s = t.compute_stats()
    return s["linear_cka"], s["rbf_cka"]


def retrieval(DXY, digits):
    n = len(digits)
    ar = torch.arange(n)
    nn_all = (DXY.argmin(1) == ar).double().mean().item()
    d = torch.from_numpy(digits)
    nn_cls = (DXY.masked_fill(d[:, None] != d[None, :], float("inf")).argmin(1) == ar
              ).double().mean().item()
    return nn_all, nn_cls


def derange_into(perm, idx, rng):
    """Write into perm a random derangement of idx (a single cycle, so pi(i) != i)."""
    order = rng.permutation(idx)
    perm[order] = np.roll(order, 1)


def null_perms(digits, rng):
    n = len(digits)
    full, cls = np.arange(n), np.arange(n)
    derange_into(full, np.arange(n), rng)
    for d in range(10):
        idx = np.flatnonzero(digits == d)
        if len(idx) > 1:
            derange_into(cls, idx, rng)
    return torch.from_numpy(cls), torch.from_numpy(full)


def score_model(net, X, Y, digits, perms, device, check=False):
    fx, fy = features(net, X, device), features(net, Y, device)
    fr = features(net, torch.rot90(X, 1, dims=(-2, -1)), device)
    cls_perm, full_perm = perms
    rows = {}
    for layer in fx:
        g = pair_grams(fx[layer], fy[layer])
        if check:  # the gram reuse must reproduce the tracker exactly
            t = EquivarianceTracker("cpu")
            t.update(fx[layer], fy[layer])
            ref = t.compute_stats()
            got = cka(g)
            assert abs(got[0] - ref["linear_cka"]) < 1e-6 and abs(got[1] - ref["rbf_cka"]) < 1e-6, \
                (layer, got, ref)
            assert all(abs(v - 1) < 1e-6 for v in cka(pair_grams(fx[layer], fx[layer]))), layer
        font, cls, shuf = cka(g), cka(g, cls_perm), cka(g, full_perm)
        rot = cka(pair_grams(fx[layer], fr[layer]))
        nn_all, nn_cls = retrieval(g["DXY"], digits)
        r = {}
        for k, (lin, rbf) in dict(font=font, cls=cls, shuf=shuf, rot90=rot).items():
            r[f"lin_{k}"], r[f"rbf_{k}"] = lin, rbf
        r["lin_delta"], r["rbf_delta"] = font[0] - cls[0], font[1] - cls[1]
        r["nn_all"], r["nn_cls"] = nn_all, nn_cls
        rows[layer] = r
    return rows


# --------------------------------------------------------------------------- output

def fmt(xs, signed=False):
    m = float(np.mean(xs))
    s = f"{m:+.4f}" if signed else f"{m:.4f}"
    return s + (f"±{np.std(xs, ddof=1):.4f}" if len(xs) > 1 else "")


def print_tables(name, res, acc):
    print(f"=== {name}: held-out-font acc {fmt(acc[name]['held'])}, "
          f"train-font acc {fmt(acc[name]['train'])} ===")
    layers = list(res[name])
    for kern, label in (("lin", "linear CKA"), ("rbf", "RBF CKA")):
        cols = ["font", "cls", "delta", "shuf", "rot90"]
        heads = ["font", "cls-shuf", "delta", "shuf", "rot90"]
        if kern == "lin":
            cols += ["nn_all", "nn_cls"]
            heads += ["NN all", "NN cls"]
        print(f"  [{label}]")
        print(f"  {'layer':<9}" + "".join(f"{h:>16}" for h in heads))
        for layer in layers:
            vals = []
            for c in cols:
                key = c if c.startswith("nn") else f"{kern}_{c}"
                vals.append(fmt(res[name][layer][key], signed=(c == "delta")))
            print(f"  {layer:<9}" + "".join(f"{v:>16}" for v in vals))
    print()


def print_summary(res, acc, n):
    print(f"=== summary (NN chance: all {1 / n:.4f}, cls ~{10 / n:.4f}) ===")
    for layer in SUMMARY_LAYERS:
        print(f"  @ {layer}")
        print(f"  {'model':<12}{'held acc':>16}{'lin font':>16}{'lin cls-shuf':>16}"
              f"{'lin delta':>16}{'rbf delta':>16}{'lin rot90':>16}{'NN cls':>16}")
        for m in MODELS:
            r = res[m][layer]
            print(f"  {m:<12}{fmt(acc[m]['held']):>16}{fmt(r['lin_font']):>16}"
                  f"{fmt(r['lin_cls']):>16}{fmt(r['lin_delta'], True):>16}"
                  f"{fmt(r['rbf_delta'], True):>16}{fmt(r['lin_rot90']):>16}"
                  f"{fmt(r['nn_cls']):>16}")
        rank = lambda key: " > ".join(sorted(MODELS, key=lambda m: -np.mean(key(m))))
        print(f"  ranking by held-out acc : {rank(lambda m: acc[m]['held'])}")
        print(f"  ranking by lin font CKA : {rank(lambda m: res[m][layer]['lin_font'])}")
        print(f"  ranking by lin delta    : {rank(lambda m: res[m][layer]['lin_delta'])}")
        print()


# --------------------------------------------------------------------------- main

def main():
    p = ArgumentParser(description="CKA under a font change (non-geometric transform).")
    p.add_argument("--font-dir", default=None, help="search here instead of system fonts")
    p.add_argument("--n-pairs", type=int, default=2000)
    p.add_argument("--train-images", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--eval-frac", type=float, default=0.3, help="fraction of fonts held out")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--min-recognised", type=int, default=7,
                   help="digits (of 10) that must match the cross-font prototype")
    p.add_argument("--no-jitter", action="store_true",
                   help="eval pairs at fixed size/position (training stays jittered)")
    p.add_argument("--quick", action="store_true", help="smoke test: small N, 1 epoch, 1 seed")
    args = p.parse_args()
    if args.quick:
        args.n_pairs, args.train_images, args.epochs, args.seeds = 400, 4000, 1, 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fonts, n_files, dropped = discover_fonts(args.font_dir, args.min_recognised)
    train_ids, held_ids = split_by_family(fonts, args.eval_frac, args.split_seed)
    print(f"{n_files} font files -> {len(fonts)} usable "
          f"({len({f['family'] for f in fonts})} families); dropped: {dropped}")
    print(f"train fonts {len(train_ids)}, held-out fonts {len(held_ids)} (split by family)")
    print(f"N = {args.n_pairs} eval pairs, {args.train_images} train images x {args.epochs} "
          f"epochs, {args.seeds} seed(s), eval jitter {'off' if args.no_jitter else 'on'}\n")
    if len(train_ids) < 2 or len(held_ids) < 2:
        sys.exit("need >= 2 train and >= 2 held-out fonts; try --font-dir")

    res = {m: defaultdict(lambda: defaultdict(list)) for m in MODELS}
    acc = {m: defaultdict(list) for m in MODELS}
    for seed in range(args.seeds):
        set_seed(seed)
        rng = np.random.default_rng(seed)
        X, Y, digits = eval_pairs(fonts, held_ids, args.n_pairs, rng, not args.no_jitter)
        perms = null_perms(digits, rng)
        Xh, yh = random_images(fonts, held_ids, 2000, rng)
        one = int(rng.choice(train_ids))
        print(f"seed {seed}: one_font = {fonts[one]['family']} {fonts[one]['style']}")
        pools = dict(untrained=None, one_font=[one], many_fonts=train_ids)
        for m in MODELS:
            net = train(fonts, pools[m], args, rng, device)
            Xt, yt = random_images(fonts, pools[m] or train_ids, 2000, rng)
            acc[m]["held"].append(accuracy(net, Xh, yh, device))
            acc[m]["train"].append(accuracy(net, Xt, yt, device))
            rows = score_model(net, X, Y, digits, perms, device,
                               check=(seed == 0 and m == "untrained"))
            for layer, r in rows.items():
                for k, v in r.items():
                    res[m][layer][k].append(v)
            print(f"  {m:<11} held-out acc {acc[m]['held'][-1]:.4f}")
    print()
    for m in MODELS:
        print_tables(m, res, acc)
    print_summary(res, acc, args.n_pairs)


if __name__ == "__main__":
    main()
