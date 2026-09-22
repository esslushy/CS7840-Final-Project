"""
DOES SEIS DETECT AN EXACTLY-EQUIVARIANT MODEL -- AND IS ITS SCORE ABOVE ITS OWN NULL?

SEIS (Lin, Farrahi & Cai, arXiv:2602.04054) scores equivariance and invariance
SEPARATELY, which is a thing no metric in this folder does. Its stated motivation
is the same distinction this repo's MI demos are built around: telling geometric
information that has been RE-ENCODED from information that has been DESTROYED.
So it deserves the same treatment every other metric here gets -- score it
against a model whose equivariance is known by construction, and report a
shuffled null beside every number.

THE METHOD, as described in the paper. Reshape an activation tensor (b,c,h,w)
so that each (sample, channel) pair is an OBSERVATION and each spatial position
is a VARIABLE, giving a (b*c) x (h*w) matrix. SVD-denoise to 99% cumulative
variance, run CCA between the clean and transformed matrices, then report

    S_equiv = (1/r) sum_i |cos(p_i, q_i)|            p_i, q_i canonical VARIATES
    S_inv   = (1/r) sum_i rho_i * |cos(w_i, v_i)|    w_i, v_i projection VECTORS

S_equiv asks whether spatial information survives up to a linear recoding.
S_inv asks whether the spatial basis itself stayed put. High equiv + low inv is
read as "re-encoded"; both low is read as "destroyed".

WHY THE AXIS CONVENTION IS THE WHOLE STORY. A rho(g) can act on two axes of a
feature tensor: it can move content around in SPACE, and it can mix CHANNELS.
SEIS puts spatial coordinates on the CCA VARIABLE axis, which CCA is free to
remix, and channels on the OBSERVATION axis, which CCA requires to correspond.
So the two halves of a representation are treated completely differently:
spatial re-encoding is free and correctly reported as re-encoding, while
channel re-encoding -- equally lossless -- is charged as lost information.
That asymmetry, not the invariance itself, is what this script measures. A
steerable network is the sharpest possible probe of it, because its rho(g) is
precisely a channel permutation composed with a spatial rotation.

WHAT IS SCORED. The architectural model is an escnn C8 steerable CNN, exactly
equivariant whether or not it is trained, which is asserted here rather than
assumed: the residual of its hidden tensor against escnn's own declared
FieldType representation is printed first and is ~1e-7. The plain CNN has no
equivariance prior. Both are run untrained (where the escnn guarantee still
holds exactly, so ground truth is unambiguous) and trained.

THE CONTROLS, which are the point of the script:

  paired      clean features vs rot90 features, the real measurement
  row-shuffle the observation pairing between the two matrices is permuted, so
              every genuine correspondence is destroyed. An honest detector
              must collapse here.
  disjoint    clean features of one set of images against rot90 features of a
              DIFFERENT set of images. No shared content at all.

MEASURED (SEED=0, 512 MNIST test images, 90-degree rotation -- a LATTICE
symmetry, so aliasing is not available as an excuse). Orientation
obs=b*c, vars=h*w; nulls are the row-shuffled control:

  model                        ground truth        S_equiv   null    S_inv
  architectural (untrained)    EXACT, 3.56e-07      0.3607  0.0217  0.1713
  architectural (trained)      EXACT, 7.17e-07      0.2238  0.0245  0.0941
  plain CNN (untrained)        no prior             0.2812  0.0249  0.1484
  plain CNN (trained, SO2 aug) no prior             0.4889  0.0243  0.1091

1. ON A PURE SPATIAL ROTATION, SEIS IS CORRECT -- state this first, because it
   is the control that makes the rest a finding rather than a misreading.
   Rotating the feature grid with channels untouched gives S_equiv = 1.0000 and
   S_inv = 0.1311, which is exactly the "re-encoded" signature the paper
   describes (its validation expects S_equiv > 0.85 under geometric transforms,
   with invariance dropping). Nothing is wrong here. S_equiv is INTENDED to be
   invariant to linear recoding, and a rotation is a permutation of the h*w
   variable axis.

2. THE DEFECT IS AN ASYMMETRY BETWEEN THE TWO AXES A rho(g) CAN ACT ON.
   Re-encoding across SPACE is free, because spatial coordinates are the CCA
   VARIABLES. Re-encoding across CHANNELS is not, because channels sit on the
   OBSERVATION axis, where CCA requires correspondence. A steerable network's
   representation permutes channels, so:

       transformation of the features   S_equiv  S_inv   SEIS says   truth
       rotate in space only              1.0000  0.1311  re-encoded  correct
       apply the model's TRUE rho(g)     0.2238  0.0941  DESTROYED   lossless

   Both are perfectly invertible; only the second touches channels. So SEIS
   calls a LOSSLESS re-encoding "information destroyed" -- which is the exact
   distinction the paper exists to draw, failing on the channel axis. This is
   the same scrambled-vs-destroyed conflation this repo documents in CKA,
   reproduced by a method built to avoid it.

3. THE RANKING INVERTS WITH TRAINING. An augmented plain CNN scores 0.4889
   against 0.2238 for a model equivariant to 7.17e-07 -- more than 2x, in the
   wrong direction, and S_inv inverts too (0.1091 vs 0.0941). This is LEE's
   failure in learned_vs_architectural_equivariance.py, reproduced by a metric
   built to avoid it. It is NOT explained by the plain CNN being trivially
   invariant: raw drift ||h(gx)-h(x)||/||h(x)|| under rho=I is 0.9602 for the
   plain CNN against 1.0607 for the architectural one, so both representations
   move comparably under rotation.

4. THE SCORE PARTLY SURVIVES DESTROYING THE PAIRING. Clean features against
   rotated features of DIFFERENT images score 0.1361 for the trained
   architectural model -- 61% of its paired 0.2238, on a 0.0240 null.

THE AXIS IS THE PAPER'S, NOT A GUESS. Section 3.2 reads "we reshape the
activations into matrices A, A' in R^(d x n), where d = h*w indexes spatial
coordinates and n = b*c indexes observations", and 3.3 denoises via the LEFT
singular vectors to "A~ = U~_A^T A in R^(k_A x n)". So observations are b*c and
the variables are the spatial coordinates -- the paper writes data matrices
features-first. That is the obs=b*c orientation above. The transposed rows this
script also prints are NOT SEIS; they are kept only as a robustness check and
no finding rests on them.

CAVEAT. The authors have released no code -- there is no repository link in the
abstract page, full text, footnotes or acknowledgements -- so this is a
reimplementation from the paper's stated equations (4) and (5) and its
Section 3.2/3.3 construction. The identity row scoring exactly 1.0000 validates
the mechanics. What cannot be validated without their code is undocumented
preprocessing: whether activations are normalized per channel, how the batch is
assembled, and whether CCA is regularized.

Note also that S_equiv reduces to the MEAN CANONICAL CORRELATION by
construction -- the cosine between canonical variates IS rho_i -- so SEIS sits
at the SVCCA rung of the invariance-group ladder analytically, not just
empirically. The paper says as much when it cites "standard practice in SVCCA"
for its truncation step.

Run: cd src/escnn_experiments && .venv/bin/python3 seis_probe_demo.py
     (add --trained to also train both models; ~25 min on CPU)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from escnn import nn as enn

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
DATA_ROOT = str(SRC.parent / "data")

from learned_vs_architectural_equivariance import (ArchitecturalCNN, LearnedCNN,
                                                   rotate_hidden_correct, train)

SEED = 0
N_IMAGES = 512          # observations become N_IMAGES * channels
VAR_KEEP = 0.99         # SEIS's stated SVD truncation
DEVICE = "cpu"


# ------------------------------------------------------------------ the metric

def _rank_for_variance(s, keep):
    """Smallest number of singular values explaining `keep` of the variance."""
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(energy, keep) + 1)


def cca(A, B, keep=VAR_KEEP):
    """CCA on SVD-denoised subspaces. Rows are observations, columns variables.
    Returns canonical correlations, variates, and projection vectors mapped back
    into the original variable space so that w_i and v_i are comparable."""
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)

    Ua, Sa, Vat = np.linalg.svd(A, full_matrices=False)
    Ub, Sb, Vbt = np.linalg.svd(B, full_matrices=False)
    ra, rb = _rank_for_variance(Sa, keep), _rank_for_variance(Sb, keep)
    Ua, Sa, Vat = Ua[:, :ra], Sa[:ra], Vat[:ra]
    Ub, Sb, Vbt = Ub[:, :rb], Sb[:rb], Vbt[:rb]

    Um, rho, Vmt = np.linalg.svd(Ua.T @ Ub)
    r = min(ra, rb)
    rho, Um, Vm = rho[:r], Um[:, :r], Vmt.T[:, :r]

    P, Q = Ua @ Um, Ub @ Vm                       # canonical variates
    W = Vat.T @ np.diag(1.0 / Sa) @ Um            # projection vectors, (n_var, r)
    V = Vbt.T @ np.diag(1.0 / Sb) @ Vm
    return rho, P, Q, W, V, ra, rb


def _mean_abs_cos(X, Y):
    num = np.abs((X * Y).sum(0))
    den = np.linalg.norm(X, axis=0) * np.linalg.norm(Y, axis=0)
    return float(np.mean(num / np.maximum(den, 1e-20)))


def seis(A, B, keep=VAR_KEEP):
    """Returns (S_equiv, S_inv, r)."""
    rho, P, Q, W, V, ra, rb = cca(A, B, keep)
    s_equiv = _mean_abs_cos(P, Q)
    num = np.abs((W * V).sum(0))
    den = np.linalg.norm(W, axis=0) * np.linalg.norm(V, axis=0)
    s_inv = float(np.mean(rho * num / np.maximum(den, 1e-20)))
    return s_equiv, s_inv, len(rho)


# ------------------------------------------------------------------ plumbing

def as_matrix(h, spatial_vars):
    """(b,c,H,W) -> 2D. spatial_vars=True puts spatial positions in the VARIABLE
    axis (observations = b*c); False transposes the two."""
    b, c, H, W = h.shape
    if spatial_vars:
        return h.reshape(b * c, H * W).double().numpy()
    return h.permute(2, 3, 0, 1).reshape(H * W, b * c).double().numpy()


def hidden(model, x):
    with torch.no_grad():
        return model(x)[1].cpu()


def report(tag, Ha, Hb, rng):
    """One model, both axis conventions, all three pairings."""
    for spatial_vars in (True, False):
        A, B = as_matrix(Ha, spatial_vars), as_matrix(Hb, spatial_vars)
        axis = "obs=b*c, vars=h*w" if spatial_vars else "obs=h*w, vars=b*c"

        e, i, r = seis(A, B)
        perm = rng.permutation(B.shape[0])
        e_s, i_s, _ = seis(A, B[perm])

        print(f"  {tag:<26s} {axis:<18s} r={r:<4d} "
              f"S_equiv={e:.4f} (null {e_s:.4f})   S_inv={i:.4f} (null {i_s:.4f})")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    te = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True,
                                    transform=transforms.ToTensor())
    xs = torch.stack([te[i][0] for i in range(2 * N_IMAGES)])
    x, x_other = xs[:N_IMAGES], xs[N_IMAGES:]
    x_rot, x_other_rot = (torch.rot90(t, 1, dims=(-2, -1)) for t in (x, x_other))

    models = [("architectural (untrained)", ArchitecturalCNN().eval(), True),
              ("plain CNN (untrained)", LearnedCNN().eval(), False)]

    if "--trained" in sys.argv:
        tr = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True,
                                        transform=transforms.ToTensor())
        loader = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=2)
        models += [("architectural (trained)", train(ArchitecturalCNN(), loader).cpu(), True),
                   ("plain CNN (trained)", train(LearnedCNN(), loader).cpu(), False)]

    for name, model, architectural in models:
        model = model.to(DEVICE)
        Hc, Hr = hidden(model, x), hidden(model, x_rot)

        gt = ""
        if architectural:
            res = ((Hr - rotate_hidden_correct(model, Hc, np.pi / 2)).norm()
                   / Hr.norm()).item()
            gt = f"  [residual under escnn's own rho(g) = {res:.2e} -> exactly equivariant]"
        print(f"\n{name}   hidden {tuple(Hc.shape)}{gt}")

        # Diagnostics that pin the scale before any verdict is read off it.
        # identity must score 1.0 or the implementation is wrong; spatial-only
        # isolates how much of the score the ROTATION costs, with no channel
        # mixing; correct-rho applies escnn's true representation to the clean
        # features, which for this model reproduces Hr exactly.
        report("identity (A vs A)", Hc, Hc, rng)
        report("spatial-only rot of feats", Hc, torch.rot90(Hc, 1, dims=(-2, -1)), rng)
        if architectural:
            report("correct rho(g) applied", Hc,
                   rotate_hidden_correct(model, Hc, np.pi / 2), rng)

        report("paired (clean vs rot90)", Hc, Hr, rng)
        report("disjoint images", Hc, hidden(model, x_other_rot), rng)


if __name__ == "__main__":
    main()
