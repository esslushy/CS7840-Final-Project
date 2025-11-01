import torch
import numpy as np
import torch.nn.functional as F

def gaussian_mi(X, Y, eps=1e-6):
    X = X.float()
    Y = Y.float()
    # X: (N, dx), Y: (N, dy)
    N = X.shape[0]
    assert N == Y.shape[0]
    XY = torch.concatenate([X, Y], axis=1)
    # center
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    XYc = XY - XY.mean(0, keepdims=True)

    # covariances (unbiased or ML? ML ok)
    # use shape (d,d)
    cov_X = (Xc.T @ Xc) / (N - 1) + eps * torch.eye(Xc.shape[1])
    cov_Y = (Yc.T @ Yc) / (N - 1) + eps * torch.eye(Yc.shape[1])
    cov_XY = (XYc.T @ XYc) / (N - 1) + eps * torch.eye(XYc.shape[1])

    # log det safely
    sign_x, logdet_x = torch.linalg.slogdet(cov_X)
    sign_y, logdet_y = torch.linalg.slogdet(cov_Y)
    sign_xy, logdet_xy = torch.linalg.slogdet(cov_XY)
    if sign_x <= 0 or sign_y <= 0 or sign_xy <= 0:
        # numerical issue: use larger eps or PCA to reduce dims
        raise ValueError("Non-positive determinant; increase eps or reduce dims")
    mi = 0.5 * (logdet_x + logdet_y - logdet_xy)
    return mi / np.log(2) # convert to bits

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]