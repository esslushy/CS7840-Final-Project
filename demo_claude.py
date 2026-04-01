import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader


def center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix: K_c = HKH where H = I - (1/n)11^T."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_kernel(X: np.ndarray) -> np.ndarray:
    return X @ X.T


def hsic(X: np.ndarray, Y: np.ndarray, unbiased: bool = True) -> float:
    """
    HSIC with linear kernel.
    X, Y: (n_samples, features) — representations of the SAME images
    from the SAME layer, but under different transforms (e.g., original vs rotated).
    """
    n = X.shape[0]
    assert X.shape[0] == Y.shape[0]

    K = linear_kernel(X)
    L = linear_kernel(Y)

    if unbiased:
        np.fill_diagonal(K, 0.0)
        np.fill_diagonal(L, 0.0)
        trace_KL = np.trace(K @ L)
        sum_K = K.sum()
        sum_L = L.sum()
        sum_KL = (K.sum(axis=1) * L.sum(axis=1)).sum()
        result = (
            trace_KL
            + (sum_K * sum_L) / ((n - 1) * (n - 2))
            - 2.0 * sum_KL / (n - 2)
        )
        return result / (n * (n - 3))
    else:
        Kc = center_kernel(K)
        Lc = center_kernel(L)
        return np.trace(Kc @ Lc) / (n * n)


def cka(X: np.ndarray, Y: np.ndarray, unbiased: bool = True) -> float:
    """
    CKA between two representation matrices.
    High CKA => the layer produces similar representations regardless of rotation
             => the layer is rotationally invariant.
    Low CKA  => the rotation substantially changes the representation
             => the layer is sensitive to rotation.
    """
    hsic_xy = hsic(X, Y, unbiased=unbiased)
    hsic_xx = hsic(X, X, unbiased=unbiased)
    hsic_yy = hsic(Y, Y, unbiased=unbiased)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return hsic_xy / denom


# ---------------------------------------------------------------------------
# Image rotation utilities
# ---------------------------------------------------------------------------

def rotate_batch(images: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rotate a batch of images by a given angle (degrees).

    Args:
        images: (N, C, H, W) tensor.
        angle: Rotation angle in degrees (counter-clockwise).

    Returns:
        Rotated images of the same shape.
    """
    theta_rad = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

    # Rotation matrix (no translation)
    theta = torch.tensor([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
    ], dtype=images.dtype).unsqueeze(0).expand(images.shape[0], -1, -1)

    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode="zeros")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Hook-based feature extractor for a PyTorch model."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self._features: Dict[str, List[np.ndarray]] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        name_to_mod = dict(self.model.named_modules())
        for name in self.layer_names:
            if name not in name_to_mod:
                avail = [n for n, _ in self.model.named_modules()][:20]
                raise ValueError(f"Layer '{name}' not found. First 20: {avail}")
            hook = name_to_mod[name].register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def fn(mod, inp, out):
            x = out.detach().cpu().numpy()
            if x.ndim >= 3:
                x = x.reshape(x.shape[0], -1)
            self._features[name].append(x)
        return fn

    def _reset(self):
        self._features = {n: [] for n in self.layer_names}

    @torch.no_grad()
    def extract(
        self, dataloader: DataLoader, device: str = "cpu", max_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        self.model.eval().to(device)
        self._reset()

        collected = 0
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            self.model(x)
            collected += x.shape[0]
            if max_samples and collected >= max_samples:
                break

        result = {}
        for name in self.layer_names:
            arr = np.concatenate(self._features[name], axis=0)
            if max_samples:
                arr = arr[:max_samples]
            result[name] = arr
        return result

    @torch.no_grad()
    def extract_with_rotation(
        self,
        dataloader: DataLoader,
        angle: float,
        device: str = "cpu",
        max_samples: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Single pass that collects features for both original and rotated images.
        Ensures the SAME images are used for both, in the SAME order.

        Returns:
            (features_original, features_rotated) — each a dict of {layer: (n, d)}
        """
        self.model.eval().to(device)

        feats_orig = {n: [] for n in self.layer_names}
        feats_rot = {n: [] for n in self.layer_names}

        collected = 0
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            # --- Original ---
            self._reset()
            self.model(x)
            for n in self.layer_names:
                feats_orig[n].append(self._features[n][0])

            # --- Rotated ---
            x_rot = rotate_batch(x, angle)
            self._reset()
            self.model(x_rot)
            for n in self.layer_names:
                feats_rot[n].append(self._features[n][0])

            collected += x.shape[0]
            if max_samples and collected >= max_samples:
                break

        def concat(d):
            return {n: np.concatenate(d[n], axis=0)[:max_samples] if max_samples
                    else np.concatenate(d[n], axis=0)
                    for n in self.layer_names}

        return concat(feats_orig), concat(feats_rot)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Full rotation invariance analysis
# ---------------------------------------------------------------------------

def rotation_invariance_analysis(
    model: nn.Module,
    layer_names: List[str],
    dataloader: DataLoader,
    angles: List[float] = [15, 30, 45, 60, 90, 135, 180],
    device: str = "cpu",
    max_samples: int = 500,
    unbiased: bool = True,
) -> Dict[str, Dict[float, Dict[str, float]]]:
    """
    For each layer and rotation angle, compute HSIC and CKA between
    representations of original vs. rotated images.

    Returns:
        {layer_name: {angle: {"hsic": float, "cka": float}}}
    """
    extractor = FeatureExtractor(model, layer_names)
    results: Dict[str, Dict[float, Dict[str, float]]] = {
        n: {} for n in layer_names
    }

    for angle in angles:
        print(f"Processing angle = {angle}°...")
        feats_orig, feats_rot = extractor.extract_with_rotation(
            dataloader, angle, device=device, max_samples=max_samples
        )

        for name in layer_names:
            X = feats_orig[name]
            Y = feats_rot[name]

            hsic_val = hsic(X, Y, unbiased=unbiased)
            cka_val = cka(X, Y, unbiased=unbiased)

            results[name][angle] = {"hsic": hsic_val, "cka": cka_val}
            print(f"  {name:>12s} | CKA = {cka_val:.4f}  HSIC = {hsic_val:.6f}")

    extractor.remove_hooks()
    return results


def print_results_table(results: Dict[str, Dict[float, Dict[str, float]]]):
    """Pretty-print the results as a table."""
    layers = list(results.keys())
    angles = sorted(next(iter(results.values())).keys())

    # Header
    angle_strs = [f"{a:>6.0f}°" for a in angles]
    print(f"\n{'CKA (original vs rotated)':^60}")
    print(f"{'Layer':<14s} " + "  ".join(angle_strs))
    print("-" * (14 + 8 * len(angles)))

    for layer in layers:
        vals = [f"{results[layer][a]['cka']:7.4f}" for a in angles]
        print(f"{layer:<14s} " + "  ".join(vals))


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torchvision.models as models
    from torch.utils.data import TensorDataset

    print("=== Rotation Invariance Analysis via HSIC/CKA ===\n")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Dummy data (replace with real images for meaningful results)
    dummy_images = torch.randn(300, 3, 224, 224)
    dummy_labels = torch.zeros(300, dtype=torch.long)
    dataset = TensorDataset(dummy_images, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    layer_names = [
        "conv1",     # very first conv
        "layer1",    # early residual block
        "layer2",    # mid
        "layer3",    # deeper
        "layer4",    # last conv block
        "avgpool",   # global avg pool (should be most invariant)
        "fc",        # classification head
    ]

    angles = [15, 30, 45, 90, 180]

    results = rotation_invariance_analysis(
        model=model,
        layer_names=layer_names,
        dataloader=dataloader,
        angles=angles,
        device="cpu",
        max_samples=300,
    )

    print_results_table(results)

    # --- What to expect with real data ---
    # Early layers (conv1, layer1): LOW CKA — features are spatially specific,
    #     rotation scrambles them.
    # Deeper layers (layer3, layer4): HIGHER CKA — features become more
    #     abstract and somewhat tolerant of rotation.
    # avgpool / fc: HIGHEST CKA — global pooling discards spatial info,
    #     so representations are more rotation-invariant.
    #
    # If you're evaluating a model trained with rotation augmentation,
    # you should see higher CKA at all layers compared to one trained without.