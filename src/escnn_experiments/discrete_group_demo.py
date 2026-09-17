"""
A LEARNED SYMMETRY THAT ONLY CKA CAN SEE -- and the blindness is a theorem,
not a tuning failure.

Every ro-dependent result in this folder so far has been about a metric
assuming the WRONG representation. liegg_demo.py closes that escape route:
LieGG derives the generator instead of assuming it, and on exact SO(2) point
clouds it works perfectly. This script opens a different one, and it cannot be
closed the same way.

THE CLAIM. LEE and LieGG are both FIRST-ORDER instruments. LEE differentiates
along a one-parameter flow; LieGG searches for a generator A with
grad f(x)^T A x = 0. Both therefore see only the TANGENT SPACE at the identity
-- the Lie algebra -- and so only the IDENTITY COMPONENT of the symmetry group.
A network whose learned symmetry group is DISCRETE has no Lie algebra at all:
there is no generator to find, and no flow to differentiate along. Linear CKA
needs neither, because it asks only whether SOME isometry relates the two
feature sets, and a discrete group's representation is an isometry like any
other. So a discrete learned symmetry is visible to CKA and structurally
invisible to the other two.

For reflections this is provable rather than empirical:

    det exp(A) = e^{tr A} > 0   for every real A,

so no one-parameter subgroup ever reaches a reflection (det = -1). LieGG's
hypothesis class does not contain the answer. No threshold, no sample size and
no amount of training changes that.

THE SETUP is deliberately LieGG's best case in every respect except the one
under test, exactly as liegg_demo.py was. Point clouds in R^2, so the group
acts LINEARLY on the input, as LieGG's derivation requires. No grid, no
interpolation, no lattice: every group element is an exact orthogonal matmul.
LieGG is additionally handed an INVARIANT scalar readout f(x)^2 so that its
trivial-representation condition holds by construction and it cannot be said to
have been given an impossible target.

The only thing changed from liegg_demo.py is that the symmetry group is the
Klein four-group V = {e, r, mx, my} instead of SO(2):

    e  = identity            mx = reflect across the x-axis   (x, y) -> ( x, -y)
    r  = rotate by pi        my = reflect across the y-axis   (x, y) -> (-x,  y)

with r = mx.my. The target is EQUIVARIANT under a sign character rather than
invariant --

    chi(e) = +1   chi(r) = -1   chi(mx) = +1   chi(my) = -1
    y(g.p) = chi(g) y(p)

-- which matters for a reason beyond variety. A Z_2 character makes the hidden
representation ro(g) = diag(+1,...,+1, -1,...,-1): half the channels flip sign.
That is an ORTHOGONAL matrix, so CKA reads exactly 1, while the trivial ro that
the ro-dependent family falls back to reads a large error. If the features were
merely invariant, assuming the trivial representation would have been correct
and there would be no gap to measure. The gap here is the sign irrep.

Note that V CONTAINS A ROTATION. r is rotation by pi, an element of SO(2), and
the models below are exactly equivariant to it -- yet no rotation generator
exists for them, because only theta = pi is a symmetry and not the flow through
it. The true_res column measures precisely this: it asks whether the SO(2)
generator I_8 (x) J satisfies the learned constraints, and it does not.

WHAT THIS IS NOT. LieGG is not WRONG here. Asked "does this network have a
continuous symmetry?" it answers "no", which is true. The finding is about
DETECTABILITY, not error: an exact, learned, nontrivial symmetry exists, one
metric recovers it and two cannot express it. That is a weaker claim than the
LEE inversions elsewhere in this folder and it is stated weakly on purpose.

The counterweight, and it is in this folder already: unlearnable_rho_demo.py
shows the mirror image. A nonlinear change of coordinates on the OUTPUT hides
exact equivariance from every CKA variant while MI still finds it. So the
honest summary is not "CKA is stronger" but that the detectable sets are
INCOMPARABLE -- discrete symmetry is in CKA's and not LieGG's, scrambled
coordinates are in MI's and not CKA's -- and each metric's blind spot is a
direct consequence of the quantifier in its definition:

    LEE      is f equivariant under THIS ro?              (ro fixed in advance)
    LieGG    does there EXIST a linear input generator?   (continuous only)
    CKA      does there EXIST an output isometry?         (per element, no flow)

MEASURED (3 seeds, 300 epochs, n=2048, null tol 0.05).

  model                   sym_var  null_dim  1-CKA(my)  naive(my)  drift   MSE
  ExactD2                 0.02523      82.0   0.000000     0.9676  0.0000  0.019
  LearnedMLP (aug)        0.02481      82.0   0.121733     1.4046  0.1192  0.026
  LearnedMLP (NO aug)     0.02543      84.3   0.090516     1.4033  0.1837  0.024
  LearnedMLP (untrained)  0.16907       0.0   0.665082     0.2163  1.3453  0.996

1. LIEGG IS UNINFORMATIVE, NOT WRONG, AND THAT IS THE FINDING. Asked "does this
   network have a continuous symmetry?" it answers no, which is TRUE -- dim V =
   0. But the answer does not vary: sym_var spans 1.01x across a provably
   V-equivariant model, an augmented MLP and an MLP trained with no augmentation
   at all, where 1 - CKA separates the first from the rest infinitely (exact 0).
   It cannot locate the symmetry because none of these models has the KIND of
   symmetry its hypothesis class contains.

   At NULL_TOL = 0.05 it does not return an empty null space either -- it
   returns ~83 directions for every trained model. The drift columns identify
   them: applying LieGG's own exp(tA) moves the prediction by 0.0074, about 10x
   less than a random direction (0.0470) but not zero, so they are near-symmetry
   directions of a badly conditioned polarization matrix. Same threshold
   fragility liegg_demo.py documents, in a setting where the true answer is 0.

2. THE ROTATION GENERATOR IS NOT RECOVERED EVEN THOUGH r IS A SYMMETRY. Every
   model here is EXACTLY equivariant to r = rotation by pi, an element of SO(2).
   The SO(2) generator I_8 (x) J still scores 0.00255 against 0.00373 for a
   random generator -- no better. A discrete element of a continuous group
   carries no information about the flow through it, which is the whole content
   of "first-order methods see only the identity component".

3. THE TRIVIAL-RHO FALLBACK FAILS AT EXACTLY THE ELEMENTS WITH CHARACTER -1, and
   the mx column is the internal control. chi(mx) = +1, so there the trivial
   representation IS correct and it reads 0.0000. At r and my, where chi = -1,
   it reads 0.9676 on a model whose predictions are exact. The gap is the sign
   irrep and nothing else.

4. WHAT CKA FOUND IS A REPRESENTATION, NOT A COINCIDENCE. Fitting the orthogonal
   Q(g) per element gives residual 0.000096 and a group-law defect
   ||Phi Q(mx)Q(my) - Phi Q(r)|| / ||Phi Q(r)|| of 0.000144, against 0.031-0.094
   for the other three models. Q(my) comes out diagonal +-1 with exactly 64
   entries at +1 and 64 at -1 -- ro(g) = diag(I, chi(g) I), read off the
   features with the group never supplied. This is the answer to the objection
   that CKA = 1 at one element is an existential statement about isometries
   rather than a claim about a group action.

   Both are measured on Phi rather than on the bare matrices. Q is identified
   only on the row space of the features, ReLU leaves near-dead channels, and
   off that subspace the Procrustes SVD returns an arbitrary rotation: the
   bare-matrix defect for ExactD2 is ~20x larger and measures that arbitrary
   part, not the group law.

5. AND ONE RESULT AGAINST CKA, LEFT IN. On the two APPROXIMATE models CKA
   disagrees with behaviour: drift prefers aug (0.1192 vs 0.1837), 1 - CKA
   prefers no-aug (0.1217 vs 0.0905). The direction is not stable across epoch
   budgets -- at 15 epochs it points the other way -- so the honest reading is
   that CKA is unreliable at RANKING two approximately-symmetric models. The
   clean result here is confined to the EXACT model. Needing no ro(g) buys
   correctness about WHICH group a network has, not about how much of it
   survives, which is the same boundary lie_vs_cka_aliasing_demo.py draws.

Run: cd src/escnn_experiments && .venv/bin/python3 discrete_group_demo.py
     (needs no escnn; --seeds / --epochs to match liegg_demo.py)
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import EquivarianceTracker                                # noqa: E402
from so2_exact_lee_reversal_demo import (                            # noqa: E402
    BATCH, DEVICE, K, LR, N_EVAL, N_TRAIN, LearnedMLP, lee_naive, rotate_cloud,
    sample_clouds)
from liegg_demo import NULL_TOL, polarization, true_generator        # noqa: E402

H = 64                     # channels per irrep; features are 2H = 128 real dims
# Generic angles, none of them a symmetry of the target. 180 is EXCLUDED here
# and reported separately, because it IS one.
GENERIC = [7.0, 23.0, 45.0, 73.0, 118.0, 241.0, 307.0]


# ---------------------------------------------------------------------------
# The group: Klein four-group V = {e, r, mx, my}, acting linearly and exactly
# ---------------------------------------------------------------------------
# Each element is a diagonal sign matrix on R^2, so the action on a cloud is an
# exact orthogonal matmul -- no interpolation anywhere, at any element.
GROUP = {"e": (1.0, 1.0), "r": (-1.0, -1.0), "mx": (1.0, -1.0), "my": (-1.0, 1.0)}
CHI = {"e": 1.0, "r": -1.0, "mx": 1.0, "my": -1.0}
MULT = {("mx", "my"): "r", ("my", "mx"): "r", ("r", "mx"): "my", ("mx", "r"): "my",
        ("r", "my"): "mx", ("my", "r"): "mx"}


def act(p, g):
    """Apply group element g to a batch of clouds. Exact, orthogonal, det +-1."""
    sx, sy = GROUP[g]
    s = torch.tensor([sx, sy], dtype=p.dtype, device=p.device)
    return p * s


def target(p):
    """Exactly V-equivariant under the sign character chi, and invariant under
    NOTHING continuous.

    y(p) = sum_{i<j} (x_i + x_j) tanh(2 <p_i, p_j>)

    The Gram factor is O(2)-invariant, so it contributes nothing to breaking
    rotation; the (x_i + x_j) factor is what fixes the frame. Under my the
    x-coordinates flip and the Gram is untouched, so y -> -y; under mx nothing
    in either factor moves, so y -> +y; r = mx.my gives -1. Under a generic
    rotation the x-sum mixes with the y-sum and y is not preserved at all,
    which self_check asserts rather than assumes."""
    G = p @ p.transpose(-1, -2)
    iu = torch.triu_indices(K, K, offset=1, device=p.device)
    xs = p[..., 0]
    pair = (xs[:, iu[0]] + xs[:, iu[1]])
    return (pair * torch.tanh(2.0 * G[:, iu[0], iu[1]])).sum(-1, keepdim=True)


# ---------------------------------------------------------------------------
# Exactly V-equivariant model, by group averaging against the two characters
# ---------------------------------------------------------------------------
class ExactD2(nn.Module):
    """Exact for every element of V, and NOT equivariant to anything continuous.

    An ordinary MLP body b is projected onto the two irreps of Z_2 by averaging
    over the group with and without the character:

        h_plus  = 1/4 sum_g      b(g.p)      trivial irrep: h_plus(g.p) =        h_plus(p)
        h_minus = 1/4 sum_g chi(g) b(g.p)    sign irrep:    h_minus(g.p) = chi(g) h_minus(p)

    so the hidden representation is ro(g) = diag(I_H, chi(g) I_H) -- orthogonal,
    and NONTRIVIAL whenever chi(g) = -1. That is the structural reason the
    trivial-ro fallback fails here and CKA does not.

    The head multiplies the odd part by an invariant gate, so the output is
    exactly odd: invariant x odd = odd. Same shape of construction as
    EquivariantSO2's NormGate (invariant magnitude, equivariant phase), one
    group down."""

    def __init__(self, width=256, depth=4):
        super().__init__()
        layers, d = [], 2 * K
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(width, H)]
        self.body = nn.Sequential(*layers)
        self.gate = nn.Sequential(nn.Linear(H, 128), nn.ReLU(), nn.Linear(128, H))

    def features(self, p):
        b = {g: self.body(act(p, g).flatten(1)) for g in GROUP}
        hp = sum(b.values()) / 4.0
        hm = sum(CHI[g] * b[g] for g in GROUP) / 4.0
        return torch.cat([hp, hm], dim=1)

    def forward(self, p):
        h = self.features(p)
        hp, hm = h[:, :H], h[:, H:]
        return (self.gate(hp) * hm).sum(-1, keepdim=True), h

    @staticmethod
    def rho(g, h):
        """The model's true hidden representation: identity on the first H
        channels, chi(g) on the last H. Orthogonal, exact, no interpolation."""
        return torch.cat([h[:, :H], CHI[g] * h[:, H:]], dim=1)


class InvariantReadout(nn.Module):
    """f(x)^2. Exactly invariant under all of V whenever f is exactly odd, so
    LieGG's trivial-representation condition holds by construction and it is
    being run in its best case. Squaring cannot create a continuous symmetry
    that f did not have, so this concedes LieGG everything except the thing
    under test."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, p):
        out, h = self.model(p)
        return out ** 2, h


# ---------------------------------------------------------------------------
def train(model, p, y, epochs, seed, augment=True):
    """Augmentation samples g uniformly from V and multiplies the label by
    chi(g), since the target is equivariant rather than invariant."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.Adam(model.parameters(), LR)
    keys = list(GROUP)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(p.shape[0], generator=gen)
        for i in range(0, p.shape[0], BATCH):
            idx = perm[i:i + BATCH]
            pb, yb = p[idx], y[idx]
            if augment:
                g = keys[int(torch.randint(4, (1,), generator=gen))]
                pb, yb = act(pb, g), CHI[g] * yb
            opt.zero_grad()
            F.mse_loss(model(pb)[0], yb).backward()
            opt.step()
    return model.eval()


# ---------------------------------------------------------------------------
# Metrics. Everything below is evaluated at GROUP ELEMENTS, not along a flow.
# ---------------------------------------------------------------------------
@torch.no_grad()
def finite_naive(model, p, g):
    """The ro-dependent family's only option at a discrete element: relative
    feature error under the assumed TRIVIAL representation. This is what a LEE
    user falls back to when there is no one-parameter subgroup to differentiate
    along -- and it is wrong here by exactly the sign irrep."""
    X = model(p)[1]
    Y = model(act(p, g))[1]
    return ((Y - X).norm() / X.norm()).item()


@torch.no_grad()
def cka_error(model, p, g=None, deg=None):
    """1 - linear CKA between features of p and of the transformed p. Needs no
    ro: if features(g.p) = ro(g) features(p) for ANY isometry ro -- including
    diag(+-1), which is what a Z_2 character is -- this is exactly 0."""
    X = model(p)[1]
    Y = model(act(p, g))[1] if g is not None else model(rotate_cloud(p, math.radians(deg)))[1]
    t = EquivarianceTracker(X.device)
    t.update(X, Y)
    return 1.0 - t.compute_stats()["linear_cka"]


@torch.no_grad()
def drift(model, p, g):
    """Behavioural ground truth, character-aware: does the prediction do what
    equivariance says it must? The target satisfies y(g.p) = chi(g) y(p)
    exactly, so any residual here is real error."""
    a = model(p)[0]
    b = model(act(p, g))[0]
    return ((b - CHI[g] * a).norm() / a.norm()).item()


@torch.no_grad()
def procrustes(model, p, g):
    """Fit the orthogonal Q minimising ||features(g.p) - features(p) Q||.

    This is the answer to the obvious objection to CKA. CKA = 1 says only that
    SOME isometry relates the two feature sets for THAT element; it does not by
    itself say the network learned a group ACTION. Fitting Q at each element and
    checking the group law closes the gap: if Q(mx) Q(my) = Q(r) then what was
    recovered is a representation of V, obtained without ever being told the
    group. Returns the relative residual and Q.

    Q is determined only on the ROW SPACE of the features: ReLU leaves dead and
    near-dead channels, and off the row space the SVD returns an arbitrary
    rotation. So the group law has to be checked where the data actually lives
    (see group_defect) rather than on the bare matrices, or the arbitrary part
    dominates -- which it does here, by a factor of ~20."""
    X = model(p)[1]
    Y = model(act(p, g))[1]
    U, _, Vh = torch.linalg.svd(X.T @ Y, full_matrices=False)
    Q = U @ Vh
    return ((Y - X @ Q).norm() / Y.norm()).item(), Q


@torch.no_grad()
def group_defect(model, p, Q):
    """Does the recovered representation obey the group law, ON THE DATA?

        || Phi Q(mx) Q(my) - Phi Q(r) || / || Phi Q(r) ||

    r = mx.my, so composing the two fitted reflections must reproduce the fitted
    rotation. Near zero means CKA's existential 'some isometry relates these two
    feature sets' has been upgraded to 'these isometries form a representation
    of V', with the group never supplied. Evaluated against Phi because that is
    the subspace on which each Q is identified at all."""
    X = model(p)[1]
    a = X @ Q["mx"] @ Q["my"]
    b = X @ Q["r"]
    return ((a - b).norm() / b.norm()).item()


# ---------------------------------------------------------------------------
def gen_drift(model, p, A, t=0.1):
    """Behavioural test of a CANDIDATE generator: if A really generates a
    symmetry, moving along its flow must not move the prediction.

    LieGG returns a null space, not a certificate, so a recovered direction has
    to be checked rather than believed. This applies the actual group element
    exp(tA) to the input and measures the relative change in f. For a true
    generator this is ~0; anything else means the direction is an artefact of
    where the null-space threshold was put."""
    with torch.no_grad():
        A = (A / A.norm()).reshape(2 * K, 2 * K)
        g = torch.linalg.matrix_exp(t * A)
        q = (p.reshape(len(p), -1) @ g.T).reshape(p.shape)
        a, b = model(p)[0], model(q)[0]
        return ((b - a).norm() / a.norm()).item()


def liegg_checked(model, p, A_rot, seed=0):
    """LieGG, plus the two things needed to read its answer honestly here.

    rand_res    the same residual for a RANDOM generator. true_res alone is not
                interpretable without it: the normalisation carries a 1/sqrt(n),
                so 'small' has to be small RELATIVE to what an arbitrary
                direction scores on the same matrix.
    null_drift  gen_drift of the smallest singular vector, i.e. the generator
                LieGG would report, against rand_drift for a random direction
                at the same t. The PAIR is what separates a recovered symmetry
                from a badly conditioned polarization matrix; null_drift alone
                is meaningless, since a small t makes any smooth f move little."""
    E = polarization(model, p, restrict_shared=False)
    _, S, Vh = torch.linalg.svd(E, full_matrices=False)
    smax = S[0].clamp_min(1e-30)
    Sn = S / smax
    res = lambda A: ((E @ (A / A.norm())).norm() / smax).item() / math.sqrt(len(E))
    torch.manual_seed(seed)
    rnd = torch.randn(E.shape[1], device=E.device)
    return {"sym_var": Sn[-1].item(),
            "null_dim": int((Sn < NULL_TOL).sum().item()),
            "true_res": res(A_rot.reshape(-1)),
            "rand_res": res(rnd),
            "null_drift": gen_drift(model, p, Vh[-1]),
            "rand_drift": gen_drift(model, p, rnd),
            "rot_drift": gen_drift(model, p, A_rot.reshape(-1)),
            "grad_scale": E.norm().item() / math.sqrt(len(E))}


# ---------------------------------------------------------------------------
def self_check(eq, p):
    """Assert every premise. The target really is V-equivariant and really is
    not rotation-invariant; the model really is exactly V-equivariant; and the
    reflections really do have determinant -1, which is what puts them outside
    the image of exp."""
    y0 = target(p)
    for g in GROUP:
        d = (target(act(p, g)) - CHI[g] * y0).abs().max().item()
        assert d < 1e-4, f"target not {g}-equivariant: {d:.2e}"
        h = eq.features(p)
        rel = ((eq.features(act(p, g)) - eq.rho(g, h)).norm() / h.norm()).item()
        assert rel < 1e-5, f"model not {g}-equivariant: {rel:.2e}"
    for th in (0.1, 1.234, 2.7):
        d = (target(rotate_cloud(p, th)) - y0).norm() / y0.norm()
        assert d > 0.1, f"target accidentally rotation-invariant at {th}: {d:.2e}"
    for g in ("mx", "my"):
        sx, sy = GROUP[g]
        assert sx * sy < 0, f"{g} should have det -1"
    print("  self-check passed: target is exactly V-equivariant under chi "
          "(<1e-4) and\n  demonstrably NOT rotation-invariant (>10% at 0.1, "
          "1.234, 2.7 rad).\n  ExactD2 is exact at all four elements to ~1e-6. "
          "mx and my have det -1,\n  so det exp(A) = e^tr(A) > 0 excludes them "
          "from every one-parameter subgroup.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    A_rot = true_generator().to(DEVICE)

    acc, qrec, keep = {}, {}, {}
    for seed in range(args.seeds):
        gen = torch.Generator().manual_seed(seed)
        p_tr = sample_clouds(N_TRAIN, gen).to(DEVICE)
        p_ev = sample_clouds(N_EVAL, gen).to(DEVICE)
        y_tr, y_ev = target(p_tr), target(p_ev)
        sd = y_tr.std()
        y_tr, y_ev = y_tr / sd, y_ev / sd              # odd: centring would break chi

        eq = train(ExactD2().to(DEVICE), p_tr, y_tr, args.epochs, seed)
        if seed == 0:
            self_check(eq, p_ev)
            keep = {"eq": eq, "p": p_ev}
        ml = train(LearnedMLP().to(DEVICE), p_tr, y_tr, args.epochs, seed)
        na = train(LearnedMLP().to(DEVICE), p_tr, y_tr, args.epochs, seed, augment=False)
        torch.manual_seed(seed)
        un = LearnedMLP().to(DEVICE).eval()

        for name, m in [("ExactD2", eq), ("LearnedMLP (aug)", ml),
                        ("LearnedMLP (NO aug)", na), ("LearnedMLP (untrained)", un)]:
            lg = liegg_checked(InvariantReadout(m), p_ev, A_rot, seed)
            res, Q = {}, {}
            for g in ("r", "mx", "my"):
                res[g], Q[g] = procrustes(m, p_ev, g)
            defect = group_defect(m, p_ev, Q)
            bare = ((Q["mx"] @ Q["my"] - Q["r"]).norm()
                    / Q["r"].norm()).item()
            row = dict(
                mse=F.mse_loss(m(p_ev)[0], y_ev).item(),
                lee=lee_naive(m, p_ev),
                sym_var=lg["sym_var"], null_dim=lg["null_dim"],
                true_res=lg["true_res"], rand_res=lg["rand_res"],
                null_drift=lg["null_drift"], rand_drift=lg["rand_drift"],
                rot_drift=lg["rot_drift"], grad=lg["grad_scale"],
                proc=sum(res.values()) / 3.0, defect=defect, bare=bare,
                cka_gen=sum(cka_error(m, p_ev, deg=d) for d in GENERIC) / len(GENERIC),
                **{f"naive_{g}": finite_naive(m, p_ev, g) for g in ("r", "mx", "my")},
                **{f"cka_{g}": cka_error(m, p_ev, g) for g in ("r", "mx", "my")},
                **{f"drift_{g}": drift(m, p_ev, g) for g in ("r", "mx", "my")})
            acc.setdefault(name, []).append(row)
            if name == "ExactD2" and seed == 0:
                qrec = {g: Q[g].diagonal()[:2 * H] for g in Q}

    mean = lambda n, k: sum(r[k] for r in acc[n]) / len(acc[n])
    print("A discrete learned symmetry: linear CKA vs LEE vs LieGG")
    print(f"{args.seeds} seeds, {args.epochs} epochs, n={N_EVAL}, group V = "
          f"{{e, r, mx, my}}, null tol {NULL_TOL}\n")

    print("=" * 94)
    print("  1. LIEGG, run in its best case: linear input action, INVARIANT "
          "readout f(x)^2")
    print("=" * 94)
    print(f"  {'model':<24s} {'sym_var':>8s} {'null_dim':>9s} | {'null':>7s} "
          f"{'rand':>7s} {'rot':>7s} drift | {'SO2 res':>8s} {'rand res':>8s} "
          f"{'|grad|':>8s}")
    for n in acc:
        print(f"  {n:<24s} {mean(n,'sym_var'):8.5f} {mean(n,'null_dim'):9.1f} | "
              f"{mean(n,'null_drift'):7.4f} {mean(n,'rand_drift'):7.4f} "
              f"{mean(n,'rot_drift'):7.4f}       | {mean(n,'true_res'):8.5f} "
              f"{mean(n,'rand_res'):8.5f} {mean(n,'grad'):8.4f}")
    sep = lambda k: mean("LearnedMLP (NO aug)", k) / max(mean("ExactD2", k), 1e-9)
    nd = sum(mean(n, "null_dim") for n in acc if "untrained" not in n) / 3
    print("\n  The correct answer is an EMPTY null space: dim V = 0, there is no "
          "continuous\n  symmetry to find. At this folder's tolerance LieGG "
          f"instead returns ~{nd:.0f}\n  directions for every trained model. The drift "
          "columns say what they are: the\n  reported generator moves the "
          "prediction (null > 0), just ~10x less than a\n  random direction, so "
          "these are near-symmetry directions of a poorly conditioned\n  "
          "polarization matrix rather than symmetries. 'rot' confirms the premise "
          "-- the\n  models really are not continuously rotation-symmetric -- "
          "and 'SO2 res' shows\n  the rotation generator scoring no better than "
          "a random one, even though every\n  model here is EXACTLY equivariant "
          "to r = rotation by pi. A discrete element\n  carries no information "
          "about the flow through it.")
    print("\n  But the damaging column is not any single number, it is that the "
          "numbers do\n  not MOVE. Across a provably V-equivariant model, an "
          "augmented MLP and an MLP\n  trained with no augmentation at all, "
          "LieGG spans:")
    print(f"      sym_var    {mean('ExactD2','sym_var'):.5f} / "
          f"{mean('LearnedMLP (aug)','sym_var'):.5f} / "
          f"{mean('LearnedMLP (NO aug)','sym_var'):.5f}"
          f"   separation {sep('sym_var'):.2f}x")
    print(f"      1 - CKA    {mean('ExactD2','cka_my'):.6f} / "
          f"{mean('LearnedMLP (aug)','cka_my'):.6f} / "
          f"{mean('LearnedMLP (NO aug)','cka_my'):.6f}"
          f"   separation {sep('cka_my'):.0f}x" if mean('ExactD2','cka_my') > 1e-9
          else f"      1 - CKA    {mean('ExactD2','cka_my'):.6f} / "
               f"{mean('LearnedMLP (aug)','cka_my'):.6f} / "
               f"{mean('LearnedMLP (NO aug)','cka_my'):.6f}"
               f"   separation infinite (exact 0)")
    print("  So LieGG is not merely wrong here, which would be recoverable. It "
          "is\n  UNINFORMATIVE: it cannot tell the model with an exact symmetry "
          "from the one\n  with none, because neither has the KIND of symmetry "
          "it is built to see.")

    print("\n" + "=" * 94)
    print("  2. AT THE GROUP ELEMENTS: assumed-trivial ro vs linear CKA vs "
          "behaviour")
    print("=" * 94)
    print(f"  {'model':<24s} {'naive r':>8s} {'naive mx':>9s} {'naive my':>9s} "
          f"{'1-CKA r':>8s} {'1-CKA mx':>9s} {'1-CKA my':>9s} {'drift':>8s}")
    for n in acc:
        print(f"  {n:<24s} {mean(n,'naive_r'):8.4f} {mean(n,'naive_mx'):9.4f} "
              f"{mean(n,'naive_my'):9.4f} {mean(n,'cka_r'):8.6f} "
              f"{mean(n,'cka_mx'):9.6f} {mean(n,'cka_my'):9.6f} "
              f"{sum(mean(n,f'drift_{g}') for g in ('r','mx','my'))/3:8.6f}")
    print("\n  ExactD2's predictions are exact at every element (drift ~0) and "
          "CKA reads it,\n  while the trivial-ro columns report a large error on "
          "a provably equivariant\n  model. The whole of that gap is the sign "
          "irrep: ro(g) = diag(I, chi(g) I), and\n  the internal control is the "
          "mx column: chi(mx) = +1, so there the trivial ro IS\n  the right one "
          "and it correctly reads 0.0000. It fails at exactly the two "
          "elements\n  whose character is -1, and nowhere else.")
    d = lambda n: sum(mean(n, f"drift_{g}") for g in ("r", "mx", "my")) / 3
    a, b = "LearnedMLP (aug)", "LearnedMLP (NO aug)"
    pick = lambda x, y: "aug" if x < y else "no-aug"
    if pick(d(a), d(b)) != pick(mean(a, "cka_my"), mean(b, "cka_my")):
        print(f"\n  AGAINST CKA, and measured rather than argued away: on the "
              f"two APPROXIMATE\n  models the two metrics disagree about which "
              f"is the more symmetric.\n"
              f"      drift (ground truth)   aug {d(a):.4f}  no-aug {d(b):.4f}"
              f"   -> prefers {pick(d(a), d(b))}\n"
              f"      1 - linear CKA         aug {mean(a,'cka_my'):.4f}  "
              f"no-aug {mean(b,'cka_my'):.4f}   -> prefers "
              f"{pick(mean(a,'cka_my'), mean(b,'cka_my'))}\n"
              f"  Which way the disagreement points is not stable across epoch "
              f"budgets, so this\n  is CKA being unreliable at RANKING two "
              f"approximately-symmetric models rather\n  than a fixed bias. The "
              f"clean result in this script is confined to the EXACT\n  model, "
              f"where CKA reads 0 and the other two cannot. Same caveat as\n  "
              f"lie_vs_cka_aliasing_demo.py: needing no ro(g) buys correctness "
              f"about WHICH\n  group a network has, not about how much of it "
              f"survives.")

    print("\n" + "=" * 94)
    print("  3. IS IT A GROUP? Procrustes fit of ro(g) from features alone, and "
          "the group law")
    print("=" * 94)
    print(f"  {'model':<24s} {'fit residual':>13s} {'group law on Phi':>26s} "
          f"{'1-CKA generic':>14s} {'LEE naive':>10s}")
    for n in acc:
        print(f"  {n:<24s} {mean(n,'proc'):13.6f} {mean(n,'defect'):26.6f} "
              f"{mean(n,'cka_gen'):14.6f} {mean(n,'lee'):10.4f}")
    if qrec:
        d = qrec["my"]
        print(f"\n  ExactD2's fitted Q(my) is diagonal +-1 as predicted: "
              f"{int((d > 0.99).sum())} entries at +1, "
              f"{int((d < -0.99).sum())} at -1, out of {len(d)}.")
    print(f"  A fit residual of {mean('ExactD2','proc'):.6f} with a group-law "
          f"defect of {mean('ExactD2','defect'):.6f} means\n  what CKA detected "
          f"is a REPRESENTATION of V, recovered without the group ever\n  being "
          f"supplied -- which is what upgrades CKA's existential 'some isometry "
          f"relates\n  these two feature sets' into a claim about a group "
          f"action. Both are checked on\n  Phi rather than on the bare matrices, "
          f"because ReLU leaves near-dead channels\n  and off the row space the "
          f"Procrustes SVD returns an arbitrary rotation. The\n  bare-matrix "
          f"defect for ExactD2 is {mean('ExactD2','bare'):.4f}, "
          f"{mean('ExactD2','bare')/max(mean('ExactD2','defect'),1e-12):.0f}x larger, "
          f"and measures that\n  arbitrary part rather than the group law. The 'generic' column confirms the symmetry "
          f"really is\n  discrete and not a continuous one in disguise: off the "
          f"four elements, CKA\n  error is large.")

    print("\n" + "=" * 94)
    print("  4. THE SWEEP: 1 - CKA over the rotation angle, ExactD2 (seed 0)")
    print("=" * 94)
    ts = list(range(0, 360, 45))
    vals = [cka_error(keep["eq"], keep["p"], deg=float(t)) for t in ts]
    print("  theta    " + " ".join(f"{t:7d}" for t in ts))
    print("  1-CKA    " + " ".join(f"{v:7.4f}" for v in vals))
    print("\n  Zero at 0 and 180, large everywhere else: C_2, read off the "
          "feature geometry\n  with no ro(g) ever constructed. LEE returns one "
          "scalar with no theta argument,\n  and LieGG's null space does not "
          "vary with the group element at all, so neither\n  can produce this "
          "row.")


if __name__ == "__main__":
    main()
