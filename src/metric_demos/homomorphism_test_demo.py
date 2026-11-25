"""
Can linear CKA test for a group HOMOMORPHISM, not just per-angle equivariance?

Yes -- but only if you pair the features the right way.

The original sweep (the 660 files in results/) scored equivariance as a STAR,
and the training scripts now use the cycle pairing below (utils.cycle_pairs).
In the star, every CKA compares the upright
features with one rotated copy,

    star_k  = linCKA( f(X), f(r^k X) )            k = 1, 2, 3

star_k = 1 says a rigid map M_k exists with f(r^k x) = M_k f(x), but only for
UPRIGHT x. Nothing ties M_1, M_2, M_3 together, so a network can pass every
star test with unrelated maps that are not a representation of C4.

The CYCLE test stacks all four rotations into one orbit-closed set and pairs
each element with its 90-degree rotation, in ONE score:

    cycle   = linCKA( [f(X), f(rX), f(r^2X), f(r^3X)],
                      [f(rX), f(r^2X), f(r^3X), f(X)] )

cycle = 1 forces a single map M with f(r y) = M f(y) for every y in the closed
set. Then M^2 f(x) = M f(rx) = f(r^2 x), M^4 = I because r^4 = id, the scale
is forced to 1 because s^4 = 1, and k -> M^k is a homomorphism C4 -> O(V).

Four synthetic "networks" whose answer is known by construction:

    equivariant conv   tanh(circular conv with the rot90-symmetric Laplacian).
                       Commutes with rot90 exactly; rho(r) = pixel permutation.
    hidden rep         f(r^k x) = g(x) R^k with R a random order-4 orthogonal
                       matrix. A genuine representation, in a scrambled basis.
    star cheater       f(r^k x) = g(x) Q_k with Q_1, Q_2, Q_3 INDEPENDENT random
                       orthogonal matrices. Exactly equivariant per angle,
                       NOT a homomorphism (Q_2 != Q_1^2).
    non-equivariant    tanh(circular conv with Sobel-x). Neither.

The ground-truth column fits the star maps M_k by least squares and reports
||M_1 M_1 - M_2|| / ||M_2||: zero iff the per-angle maps compose like rotations.

Measured output (seed 0, n = 400 images of 8x8, +-1 pixels):

    network              star90  star180 star270    cycle   |M1M1-M2|/|M2|  homomorphism
    equivariant conv     1.0000   1.0000  1.0000   1.0000           0.0000  yes
    hidden rep           1.0000   1.0000  1.0000   1.0000           0.0000  yes
    star cheater         1.0000   1.0000  1.0000   0.3418           1.4166  NO
    non-equivariant      0.2592   1.0000  0.2592   0.2227           0.8900  NO

Read the third row: the star test -- what the original sweep computed -- gives the
cheater a perfect score at every angle. The cycle test, using the SAME four
forward passes, drops to 0.34. It agrees with the ground truth on all four rows.

The non-equivariant row's star180 = 1.0000 is real, not a bug: Sobel-x rotated
by 180 degrees is -Sobel-x, and tanh is odd, so f(r^2 x) = -P^2 f(x) exactly.
The network IS equivariant to the subgroup C2 = {e, r^2}, just not to C4.

Caveat: this is the exact case. cycle = 1 certifies a homomorphism; there is
no theorem that cycle = 0.95 means "approximately a homomorphism".
"""
import numpy as np

SEED = 0
N = 400            # upright images; the closed set has 4N
SIDE = 8
D_HIDDEN = 32      # feature width for the two basis-defined networks

LAPLACIAN = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    K, L = X @ X.T, Y @ Y.T
    return float((K * L).sum() / np.sqrt((K * K).sum() * (L * L).sum()))


def circ_conv(images, kernel):
    out = np.zeros(images.shape)
    kh, kw = kernel.shape
    for di in range(kh):
        for dj in range(kw):
            out += kernel[di, dj] * np.roll(images, (di - kh // 2, dj - kw // 2), axis=(-2, -1))
    return out


def random_orthogonal(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def order4_orthogonal(d, rng):
    """A random orthogonal R with R^4 = I: 90-degree rotations on 2x2 blocks,
    conjugated into a random basis."""
    block = np.array([[0.0, -1.0], [1.0, 0.0]])
    R = np.kron(np.eye(d // 2), block)
    B = random_orthogonal(d, rng)
    return B @ R @ B.T


def build_networks(images, rng):
    """Each entry maps to [F_0, F_1, F_2, F_3], F_k = features of r^k X."""
    rots = [np.rot90(images, k, axes=(-2, -1)) for k in range(4)]
    flat = lambda a: a.reshape(len(a), -1)

    W = rng.standard_normal((SIDE * SIDE, D_HIDDEN)) / SIDE
    g = np.tanh(flat(images) @ W)              # arbitrary features of upright x

    R = order4_orthogonal(D_HIDDEN, rng)
    Qs = [np.eye(D_HIDDEN)] + [random_orthogonal(D_HIDDEN, rng) for _ in range(3)]

    return {
        "equivariant conv": [flat(np.tanh(circ_conv(x, LAPLACIAN))) for x in rots],
        "hidden rep":       [g @ np.linalg.matrix_power(R, k) for k in range(4)],
        "star cheater":     [g @ Qs[k] for k in range(4)],
        "non-equivariant":  [flat(np.tanh(circ_conv(x, SOBEL_X))) for x in rots],
    }


def fit_map(A, B):
    """Least-squares linear M with (B - mean) ~ (A - mean) M."""
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    return np.linalg.lstsq(Ac, Bc, rcond=None)[0]


def main():
    rng = np.random.default_rng(SEED)
    images = rng.choice([-1.0, 1.0], size=(N, SIDE, SIDE))
    nets = build_networks(images, rng)

    print(f"n = {N} upright {SIDE}x{SIDE} images; closed set = {4 * N} "
          f"(all four rotations of each)\n")
    print(f"{'network':<19s} {'star90':>7s} {'star180':>8s} {'star270':>7s} "
          f"{'cycle':>8s}   {'|M1M1-M2|/|M2|':>14s}  homomorphism")

    for name, F in nets.items():
        star = [linear_cka(F[0], F[k]) for k in (1, 2, 3)]
        cycle = linear_cka(np.vstack(F), np.vstack(F[1:] + F[:1]))

        M1, M2 = fit_map(F[0], F[1]), fit_map(F[0], F[2])
        comp_err = np.linalg.norm(M1 @ M1 - M2) / np.linalg.norm(M2)

        print(f"{name:<19s} {star[0]:7.4f} {star[1]:8.4f} {star[2]:7.4f} "
              f"{cycle:8.4f}   {comp_err:14.4f}  {'yes' if comp_err < 1e-6 else 'NO'}")

    print("\nstar  = what the original sweep computed (upright vs each rotation, separately)")
    print("cycle = one CKA on the orbit-closed set, each element paired with its 90-degree rotation")


if __name__ == "__main__":
    main()
