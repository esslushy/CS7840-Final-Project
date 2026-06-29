import numpy as np
import torch
class Random90Rotation:
    def __call__(self, img):
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(img, k, dims=(-2, -1))
    
def linear_renyi2_nmi(L_base, L_transformed):
    """
    Computes Centered Linear Rényi 2 Normalized Mutual Information normalized by Joint Entropy
    across a full dataset simultaneously. 
    
    L_base: Full dataset activations for normal input. Shape: (N, Dimensions) or (N, C, H, W)
    L_transformed: Full dataset activations for transformed input. Shape: (N, Dimensions) or (N, C, H, W)
    """
    # 1. Ensure inputs are float32/64 tensors and match device
    device = L_base.device
    
    # 2. Flatten spatial dimensions automatically if handling grids (Fluids, Images)
    if L_base.dim() > 2:
        L_base = L_base.flatten(start_dim=1)
        L_transformed = L_transformed.flatten(start_dim=1)
        
    N = L_base.size(0)
    
    # 3. Force Global Dataset Mean Centering
    # This completely strips out background bias and global layer offsets
    A_centered = L_base - torch.mean(L_base, dim=0, keepdim=True)
    B_centered = L_transformed - torch.mean(L_transformed, dim=0, keepdim=True)

    # 4. Compute Full Global Linear Gram Matrices (N x N)
    K_x = torch.matmul(A_centered, A_centered.T)
    K_y = torch.matmul(B_centered, B_centered.T)

    # 5. Extract Full Global Trace Denominators
    tr_kx = torch.trace(K_x)
    tr_ky = torch.trace(K_y)
    
    # Guard against completely dead or unvarying layers
    if tr_kx == 0 or tr_ky == 0:
        return 0.0

    # 6. Normalize Gram Matrices to form proxy probability densities
    A_norm = K_x / tr_kx
    B_norm = K_y / tr_ky

    # 7. Compute Global Dataset Traces 
    # Using element-wise multiplication sum runs in O(N^2) instead of O(N^3)
    global_tr_A2 = torch.sum(A_norm * A_norm)
    global_tr_B2 = torch.sum(B_norm * B_norm)
    global_tr_AB = torch.sum(A_norm * B_norm)

    # 8. Compute Marginal Linear Entropies
    # Small 1e-12 epsilon keeps logs stable
    H2_A = -torch.log2(global_tr_A2 + 1e-12)
    H2_B = -torch.log2(global_tr_B2 + 1e-12)
    
    if H2_A <= 0 or H2_B <= 0:
        return 0.0

    # 9. Calculate Absolute Shared Bits
    mi_absolute = torch.log2(global_tr_AB / (global_tr_A2 * global_tr_B2) + 1e-12)
    mi_absolute = torch.clamp(mi_absolute, min=0.0)

    # 10. Symmetric Uncertainty Normalization
    symmetric_mi = (2.0 * mi_absolute) / (H2_A + H2_B)
    
    # Strict 0.0 to 1.0 bounding guard clip
    return torch.clamp(symmetric_mi, min=0.0, max=1.0).item()


class UnifiedEquivarianceTracker:
    def __init__(self, device="cpu"):
        """
        A unified, hyperparameter-free metric for tracking learned equivariance.
        Uses Centered Linear Rényi 2 Mutual Information normalized by Joint Entropy.
        """
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        """Resets the accumulated mini-batch statistics."""
        self.total_samples = 0
        self.sum_tr_A2 = 0.0
        self.sum_tr_B2 = 0.0
        self.sum_tr_AB = 0.0

    @torch.no_grad()
    def update(self, L_base, L_transformed):
        """
        Accumulates cross-batch statistics.
        L_base: Activations from the un-transformed input (Batch_Size, Dimensions)
        L_transformed: Activations from the transformed input (Batch_Size, Dimensions)
        """
        L_base = L_base.to(self.device, dtype=torch.float32)
        L_transformed = L_transformed.to(self.device, dtype=torch.float32)
        
        # Flatten spatial structures if treating grids natively (e.g., Fluids/Pixels)
        if L_base.dim() > 2:
            L_base = L_base.flatten(start_dim=1)
            L_transformed = L_transformed.flatten(start_dim=1)
            
        batch_size = L_base.size(0)

        # 1. Force Batch Centering (Essential for hyperparameter-free linear representations)
        A_centered = L_base - torch.mean(L_base, dim=0, keepdim=True)
        B_centered = L_transformed - torch.mean(L_transformed, dim=0, keepdim=True)

        # 2. Compute Linear Gram Matrices
        K_x = torch.matmul(A_centered, A_centered.T)
        K_y = torch.matmul(B_centered, B_centered.T)

        # 3. Trace Normalization
        tr_kx = torch.trace(K_x)
        tr_ky = torch.trace(K_y)
        
        if tr_kx == 0 or tr_ky == 0:
            return  # Guard against completely dead layers

        A_norm = K_x / tr_kx
        B_norm = K_y / tr_ky

        # 4. Compute O(N^2) Trace Metrics via Element-wise products
        batch_tr_A2 = torch.sum(A_norm * A_norm).item()
        batch_tr_B2 = torch.sum(B_norm * B_norm).item()
        batch_tr_AB = torch.sum(A_norm * B_norm).item()

        # 5. Accumulate weighted statistics to support uneven final loader batches
        self.sum_tr_A2 += batch_tr_A2 * batch_size
        self.sum_tr_B2 += batch_tr_B2 * batch_size
        self.sum_tr_AB += batch_tr_AB * batch_size
        self.total_samples += batch_size

    def compute(self):
        """
        Computes the Symmetric Uncertainty ratio.
        Returns a value strictly bounded between 0.0 and 1.0.
        """
        if self.total_samples == 0:
            return 0.0

        # Extract global dataset traces from accumulators
        global_tr_A2 = self.sum_tr_A2 / self.total_samples
        global_tr_B2 = self.sum_tr_B2 / self.total_samples
        global_tr_AB = self.sum_tr_AB / self.total_samples

        # 1. Compute marginal linear entropies
        H2_A = -torch.log2(torch.tensor(global_tr_A2 + 1e-12)).item()
        H2_B = -torch.log2(torch.tensor(global_tr_B2 + 1e-12)).item()
        
        if H2_A <= 0 or H2_B <= 0:
            return 0.0

        # 2. Compute absolute shared bits
        mi_absolute = torch.log2(torch.tensor(global_tr_AB / (global_tr_A2 * global_tr_B2) + 1e-12)).item()
        mi_absolute = max(0.0, mi_absolute)

        # 3. Apply Symmetric Uncertainty Normalization
        symmetric_mi = (2.0 * mi_absolute) / (H2_A + H2_B)
        
        # Safe numeric bounding guard clip
        return min(1.0, max(0.0, symmetric_mi))
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tracking on device: {device}\n" + "="*40)
    
    # Generate mock network representations (1000 items, with explicit global mean shifts)
    # Adding a massive arbitrary bias (+15.0) to simulate raw uncentered activations
    X_data = torch.randn(1000, 64) + 15.0  
    Y_data = X_data[:, :8] @ torch.randn(8, 128) + torch.randn(1000, 128) * 0.1 + 15.0
    
    dataset = TensorDataset(X_data, Y_data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    print(f"Unbatched:  {linear_renyi2_nmi(X_data, Y_data)}")
    metric = UnifiedEquivarianceTracker(device)
    for x,y in dataloader:
        metric.update(x, y)
    print(f"Batched:    {metric.compute()}")