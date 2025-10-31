import torch

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
    return mi  # nats