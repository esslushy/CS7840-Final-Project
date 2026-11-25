"""
Published datasets for the flow and elasticity tasks, downloaded on first use.

Every loader here does the same three things:

1. **Fetch** the raw files into ``DATA_ROOT/<dataset>/`` (``./data`` relative to
   the working directory, like torchvision, or ``$EQUIV_DATA_DIR`` if set).
2. **Preprocess** once: transpose to this repo's layout, downsample, pick
   particle subsets, and normalise with rotation-invariant constants.
3. **Cache** the result as one ``.pt`` file, so later runs load in seconds.

The cache is built under an exclusive lock file. Ten SLURM array tasks starting
together therefore build it once; the other nine wait and then load it.

Layout conventions, which the rotation functions in the task scripts rely on:

- Grid fields are ``(C, H, W)`` with **rows indexing y (increasing) and columns
  indexing x**. Both PDEArena and PDEBench store ``(..., x, y)``, so they are
  transposed here. The axis order was checked against the physics. In PDEArena
  the divergence-free condition pairs ``vx`` with axis 2, and the smoke centroid
  rises along axis 3. In PDEBench, mass conservation holds best with ``Vx`` on
  axis 2.
- Vector channels come first (``vx, vy``), scalar channels after.
- Normalisation never treats vector components separately. Both components
  share one scale and no mean is subtracted, because a per-component affine map
  does not commute with rotation. Scalar channels are standardised freely.
- Particle positions are centred on the simulation box, so rotation about the
  origin is rotation about the box centre.

``python published_data.py`` downloads and preprocesses everything up front,
which is worth doing once on a login node before submitting the sweep.
"""

import hashlib
import io
import json
import os
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

DATA_ROOT = Path(os.environ.get("EQUIV_DATA_DIR", "data"))

# Bump when preprocessing changes, so stale caches are rebuilt rather than reused.
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Download / cache plumbing
# ---------------------------------------------------------------------------

def _atomic_torch_save(obj, path):
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@contextmanager
def _exclusive(lock_path, stale_after=6 * 3600):
    """Hold an exclusive lock file for the duration of the block.

    A lock older than ``stale_after`` seconds is assumed to be left over from a
    killed job and is broken.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    announced = False
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > stale_after:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if not announced:
                print(f"[data] waiting for another process to finish {lock_path.stem} ...", flush=True)
                announced = True
            time.sleep(10)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _cached(name, build):
    """Load ``DATA_ROOT/cache/<name>.pt``, building it with ``build()`` if absent."""
    path = DATA_ROOT / "cache" / f"{name}_v{CACHE_VERSION}.pt"
    if not path.exists():
        with _exclusive(path.with_suffix(".lock")):
            if not path.exists():           # another process may have built it meanwhile
                print(f"[data] building {path} (one-time) ...", flush=True)
                t0 = time.time()
                _atomic_torch_save(build(), path)
                print(f"[data] built {path.name} in {time.time() - t0:.0f}s", flush=True)
    return torch.load(path, weights_only=True)


def _md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _download(url, dest, md5=None, retries=5):
    """Stream ``url`` to ``dest`` (resuming a partial ``.part`` file), then verify ``md5``."""
    import requests

    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:          # already complete
                    break
                r.raise_for_status()
                mode = "ab" if r.status_code == 206 else "wb"
                total = int(r.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0)
                print(f"[data] downloading {dest.name} ({total / 1e6:.0f} MB) ...", flush=True)
                with open(part, mode) as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            break
        except Exception as e:  # network hiccup: retry, resuming from what we have
            if attempt == retries - 1:
                raise
            print(f"[data] download error ({e}); retrying", flush=True)
            time.sleep(5 * (attempt + 1))
    if md5 is not None and _md5(part) != md5:
        part.unlink()
        raise RuntimeError(f"MD5 mismatch for {dest.name}; deleted the partial file, re-run to retry")
    os.replace(part, dest)
    return dest


class _HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests.

    Lets h5py read the metadata of a remote HDF5 file without downloading it.
    """

    def __init__(self, url):
        import requests
        self._session = requests.Session()
        r = self._session.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
        r.raise_for_status()
        self.url = r.url               # follow redirects once (DaRUS -> signed S3 URL)
        self.size = int(r.headers["Content-Range"].split("/")[-1])
        r.close()
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read_range(self, start, length, retries=5):
        for attempt in range(retries):
            try:
                r = self._session.get(self.url, timeout=300,
                                      headers={"Range": f"bytes={start}-{start + length - 1}"})
                r.raise_for_status()
                if len(r.content) != length:
                    raise IOError(f"short read: {len(r.content)} of {length} bytes")
                return r.content
            except Exception as e:
                if attempt == retries - 1:
                    raise
                print(f"[data] range read error ({e}); retrying", flush=True)
                time.sleep(5 * (attempt + 1))

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.read_range(self.pos, n)
        b[:n] = data
        self.pos += n
        return n


def _pool(x, grid_size):
    """Average-pool the trailing two (square) dims down to ``grid_size``.

    Average pooling by an integer factor commutes exactly with rot90, so it
    does not disturb the equivariance of the data.
    """
    factor = x.shape[-1] // grid_size
    assert x.shape[-1] == x.shape[-2] == factor * grid_size, (x.shape, grid_size)
    if factor == 1:
        return x
    lead = x.shape[:-3]
    x = x.reshape(-1, *x.shape[-3:])
    return Fn.avg_pool2d(x, factor).reshape(*lead, x.shape[-3], grid_size, grid_size)


def _grid_norm_stats(traj, n_vector=2):
    """Rotation-safe normalisation constants for (N, T, C, H, W) trajectories.

    Vector channels share one RMS scale and keep zero offset; scalar channels
    get their own mean and std.
    """
    C = traj.shape[2]
    mean = torch.zeros(C)
    scale = torch.ones(C)
    scale[:n_vector] = traj[:, :, :n_vector].pow(2).mean().sqrt()
    for c in range(n_vector, C):
        mean[c] = traj[:, :, c].mean()
        scale[c] = traj[:, :, c].std()
    return mean, scale


def _apply_grid_norm(traj, mean, scale):
    return (traj - mean.view(1, 1, -1, 1, 1)) / scale.view(1, 1, -1, 1, 1)


# ---------------------------------------------------------------------------
# PDEArena NavierStokes-2D: PhiFlow smoke plume with buoyancy (fluid_flow, buoyant)
# ---------------------------------------------------------------------------
#
# Gupta & Brandstetter 2022, "Towards Multi-spatiotemporal-scale Generalized PDE
# Modeling". Incompressible Navier-Stokes plus an advected smoke density, with
# buoyancy along +y, in a closed 32 x 32 box on a 128 x 128 grid. Each file holds
# 25 trajectories of 14 stored steps, with keys u (smoke), vx, vy in (traj, t, x, y).

PDEARENA_REPO = "pdearena/NavierStokes-2D"


def _pdearena_build(split, n_files, grid_size):
    import h5py
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(PDEARENA_REPO, repo_type="dataset")
                   if f.startswith(f"NavierStokes2D_{split}_") and f.endswith(".h5"))
    if len(files) < n_files:
        raise ValueError(f"PDEArena has only {len(files)} {split} files, asked for {n_files}")
    out = []
    for fname in files[:n_files]:
        path = hf_hub_download(PDEARENA_REPO, fname, repo_type="dataset",
                               local_dir=DATA_ROOT / "pdearena_ns2d")
        with h5py.File(path, "r") as f:
            g = f[split]
            # (traj, t, x, y) -> (traj, t, y, x); channels (vx, vy, smoke)
            fields = [torch.from_numpy(g[k][:].astype(np.float32)).transpose(-1, -2)
                      for k in ("vx", "vy", "u")]
        out.append(_pool(torch.stack(fields, dim=2), grid_size))
    return {"traj": torch.cat(out)}


def load_pdearena_smoke(split, n_files, grid_size):
    """Normalised (N_traj, T, 3, G, G) trajectories, channels (vx, vy, smoke)."""
    def get(s, n):
        return _cached(f"pdearena_smoke_{s}_{n}f_{grid_size}",
                       lambda: _pdearena_build(s, n, grid_size))["traj"]
    train = get("train", n_files["train"])
    mean, scale = _grid_norm_stats(train)
    traj = train if split == "train" else get(split, n_files[split])
    return _apply_grid_norm(traj, mean, scale)


# ---------------------------------------------------------------------------
# PDEBench 2D CFD: compressible Navier-Stokes, periodic, random ICs (fluid_flow, isotropic)
# ---------------------------------------------------------------------------
#
# Takamoto et al. 2022, "PDEBench". 2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128:
# 10,000 trajectories of 21 steps on a periodic 128 x 128 unit square, with no
# forcing, so the dynamics are exactly symmetric under 90-degree rotations.
# The file is 55 GB, but its datasets are contiguous and uncompressed, so the
# trajectories needed are read straight out of it with HTTP range requests.
# Train uses the first trajectories and test the last, so they never overlap.

PDEBENCH_CFD_URL = "https://darus.uni-stuttgart.de/api/access/datafile/164688"
PDEBENCH_CFD_KEYS = ("Vx", "Vy", "density", "pressure")   # vector first, then scalars


def _pdebench_build(split, n_traj, grid_size, block=25):
    import h5py

    remote = _HTTPRangeFile(PDEBENCH_CFD_URL)
    with h5py.File(remote, "r") as f:
        meta = {}
        for k in PDEBENCH_CFD_KEYS:
            d = f[k]
            assert d.chunks is None and d.compression is None, f"{k} is not contiguous"
            meta[k] = (d.id.get_offset(), d.shape, d.dtype)
    N, T, X, Y = meta["Vx"][1]
    start = 0 if split == "train" else N - n_traj
    per_traj = T * X * Y
    out = []
    for lo in range(start, start + n_traj, block):
        hi = min(lo + block, start + n_traj)
        fields = []
        for k in PDEBENCH_CFD_KEYS:
            offset, _, dtype = meta[k]
            raw = remote.read_range(offset + lo * per_traj * dtype.itemsize,
                                    (hi - lo) * per_traj * dtype.itemsize)
            a = np.frombuffer(raw, dtype=dtype).reshape(hi - lo, T, X, Y).astype(np.float32)
            fields.append(torch.from_numpy(a).transpose(-1, -2))   # (x, y) -> (y, x)
        out.append(_pool(torch.stack(fields, dim=2), grid_size))
        print(f"[data] pdebench_cfd {split}: {hi - start}/{n_traj} trajectories", flush=True)
    return {"traj": torch.cat(out)}


def load_pdebench_cfd(split, n_traj, grid_size):
    """Normalised (N_traj, T, 4, G, G) trajectories, channels (vx, vy, density, pressure)."""
    def get(s, n):
        return _cached(f"pdebench_cfd_{s}_{n}t_{grid_size}",
                       lambda: _pdebench_build(s, n, grid_size))["traj"]
    train = get("train", n_traj["train"])
    mean, scale = _grid_norm_stats(train)
    traj = train if split == "train" else get(split, n_traj[split])
    return _apply_grid_norm(traj, mean, scale)


# ---------------------------------------------------------------------------
# LagrangeBench 2D SPH particle flows (fluid_flow_particles)
# ---------------------------------------------------------------------------
#
# Toshev et al. 2023, "LagrangeBench" (NeurIPS Datasets & Benchmarks).
#   tgv: Taylor-Green vortex, periodic unit square, no external force (isotropic)
#   dam: dam break under gravity, walled box (buoyant / gravity-broken)
# The files store positions only, as (T, P, 2) per trajectory, every 100 solver
# steps. Velocity is the displacement between stored frames. On periodic
# domains it uses the shortest wrapped displacement, and next positions are
# unwrapped relative to the current ones.

LAGRANGEBENCH_ZENODO = "https://zenodo.org/api/records/10491868/files/{}.zip/content"
LAGRANGEBENCH_CASES = {
    "tgv": "2D_TGV_2500_10kevery100",
    "dam": "2D_DAM_5740_20kevery100",
}
LAGRANGEBENCH_FLUID_TYPE = 0   # particle_type of fluid particles; walls are static


def _lagrangebench_dir(case):
    name = LAGRANGEBENCH_CASES[case]
    root = DATA_ROOT / "lagrangebench"
    folder = root / name
    if not (folder / "metadata.json").exists():
        with _exclusive(root / f"{name}.lock"):
            if not (folder / "metadata.json").exists():
                z = _download(LAGRANGEBENCH_ZENODO.format(name), root / f"{name}.zip")
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(root)
    return folder


def _lagrangebench_build(case, split, num_particles, seed=1234):
    import h5py

    folder = _lagrangebench_dir(case)
    meta = json.loads((folder / "metadata.json").read_text())
    bounds = np.asarray(meta["bounds"], dtype=np.float64)          # (2, 2): [[x0, x1], [y0, y1]]
    periodic = np.asarray(meta["periodic_boundary_conditions"][:2], dtype=bool)
    center = bounds.mean(axis=1)
    box = bounds[:, 1] - bounds[:, 0]
    pos_scale = 2.0 / box.max()          # one isotropic scale: longest side -> [-1, 1]

    # A fixed generator, independent of the training seed, gives every run the same data.
    rng = np.random.default_rng(seed)
    states_t, states_tp1 = [], []
    with h5py.File(folder / f"{split}.h5", "r") as f:
        for key in sorted(f.keys()):
            pos = f[key]["position"][:].astype(np.float64)              # (T, P, 2)
            fluid = np.flatnonzero(f[key]["particle_type"][:] == LAGRANGEBENCH_FLUID_TYPE)
            disp = pos[1:] - pos[:-1]                                    # (T-1, P, 2)
            disp[..., periodic] -= box[periodic] * np.round(disp[..., periodic] / box[periodic])
            # pair (t, t+1) for t = 1 .. T-2: velocity at t needs frame t-1
            for t in range(1, pos.shape[0] - 1):
                idx = rng.choice(fluid, size=num_particles, replace=False)
                p_t = (pos[t, idx] - center) * pos_scale
                v_t = disp[t - 1, idx] * pos_scale
                p_tp1 = p_t + disp[t, idx] * pos_scale                   # unwrapped next position
                v_tp1 = disp[t, idx] * pos_scale
                states_t.append(np.concatenate([p_t, v_t], -1))
                states_tp1.append(np.concatenate([p_tp1, v_tp1], -1))
    return {"state_t": torch.from_numpy(np.stack(states_t)).float(),
            "state_tp1": torch.from_numpy(np.stack(states_tp1)).float()}


def load_lagrangebench(case, split, num_particles):
    """(N, P, 4) = (x, y, vx, vy) pairs at t and t+1, positions centred on the box.

    Velocities are rescaled by one rotation-invariant scalar (from train) so
    they have unit RMS, which keeps them on the same footing as positions in
    the MSE loss.
    """
    def get(s):
        return _cached(f"lagrangebench_{case}_{s}_{num_particles}p",
                       lambda: _lagrangebench_build(case, s, num_particles))
    train = get("train")
    v_gain = 1.0 / train["state_t"][..., 2:4].pow(2).mean().sqrt()
    data = train if split == "train" else get(split)
    gain = torch.tensor([1.0, 1.0, v_gain, v_gain])
    return data["state_t"] * gain, data["state_tp1"] * gain


# ---------------------------------------------------------------------------
# Geo-FNO Elasticity point clouds (stress_prediction_particles)
# ---------------------------------------------------------------------------
#
# Li et al. 2022, "Fourier Neural Operator with Learned Deformations for PDEs on
# General Geometries". A hyperelastic unit cell with a random central void,
# clamped at the bottom with tension on the top edge. Each sample has 972 mesh
# points with one scalar stress per point. The fixed loading direction breaks
# rotation symmetry. Split as in the Geo-FNO script: the first 1000 samples
# are train and the last 200 are test.

GEOFNO_DRIVE_IDS = {
    "Random_UnitCell_XY_10.npy": "1I-fO-RsFvD3nqBuFrg67R0yqTFdD_gpA",      # (972, 2, 2000)
    "Random_UnitCell_sigma_10.npy": "1Ia5izgUum-IQLdO6PW70HO8AdAqA_IVb",   # (972, 2000)
}
GEOFNO_SPLITS = {"train": slice(0, 1000), "test": slice(-200, None)}


def _geofno_files():
    import gdown

    root = DATA_ROOT / "geofno_elasticity"
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fid in GEOFNO_DRIVE_IDS.items():
        p = root / name
        if not p.exists():
            with _exclusive(root / f"{name}.lock"):
                if not p.exists():
                    print(f"[data] downloading {name} from Google Drive ...", flush=True)
                    tmp = root / f"{name}.part{os.getpid()}"
                    gdown.download(id=fid, output=str(tmp), quiet=True)
                    os.replace(tmp, p)
        paths[name] = p
    return paths


def _geofno_build(split, num_particles, subsets, seed=1234):
    paths = _geofno_files()
    xy = np.load(paths["Random_UnitCell_XY_10.npy"]).transpose(2, 0, 1)       # (2000, 972, 2)
    sigma = np.load(paths["Random_UnitCell_sigma_10.npy"]).T                   # (2000, 972)
    xy, sigma = xy[GEOFNO_SPLITS[split]], sigma[GEOFNO_SPLITS[split]]
    rng = np.random.default_rng(seed)
    pos, out = [], []
    for i in range(xy.shape[0]):
        for _ in range(subsets):
            idx = rng.choice(xy.shape[1], size=num_particles, replace=False)
            pos.append((xy[i, idx] - 0.5) * 2.0)       # unit cell -> [-1, 1], centred
            out.append(sigma[i, idx, None])
    return {"pos": torch.from_numpy(np.stack(pos)).float(),
            "stress": torch.from_numpy(np.stack(out)).float()}


def load_geofno_elasticity(split, num_particles, subsets):
    """(N, P, 2) centred positions and (N, P, 1) stress, scaled by the train RMS.

    Each geometry contributes ``subsets[split]`` random point subsets, so the
    sample count is (geometries x subsets).
    """
    def get(s):
        return _cached(f"geofno_elasticity_{s}_{num_particles}p_{subsets[s]}s",
                       lambda: _geofno_build(s, num_particles, subsets[s]))
    train = get("train")
    scale = train["stress"].pow(2).mean().sqrt()
    data = train if split == "train" else get(split)
    return data["pos"], data["stress"] / scale


# ---------------------------------------------------------------------------
# Prefetch everything
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Uses each task script's own sizes, so the caches match what training loads.
    import fluid_flow
    import fluid_flow_particles
    import stress_prediction_particles

    for ds in fluid_flow.DATASETS:
        for split in ("train", "test"):
            fluid_flow.make_dataset(ds, split, rotate=False)
    for ds in fluid_flow_particles.DATASETS:
        for split in ("train", "test"):
            fluid_flow_particles.make_dataset(ds, split, rotate=False)
    for ds in stress_prediction_particles.DATASETS:
        for split in ("train", "test"):
            stress_prediction_particles.make_dataset(ds, split, rotate=False)
    print(f"[data] all published datasets cached under {DATA_ROOT.resolve()}")
