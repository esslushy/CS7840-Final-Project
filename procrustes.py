import torch
from scipy.spatial import procrustes

if __name__ == "__main__":
    # Vectors
    vectors = torch.randn(100, 3)

    u1, u2, u3 = torch.rand(3)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    # Quaternion to rotation matrix
    R = torch.tensor([
        [1 - 2*(q3**2 + q4**2),     2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
        [    2*(q2*q3 + q1*q4), 1 - 2*(q2**2 + q4**2),     2*(q3*q4 - q1*q2)],
        [    2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2), 1 - 2*(q2**2 + q3**2)]
    ])

    rotated_vectors = vectors @ R.T

    print(procrustes(vectors, vectors)[2])
    print(procrustes(vectors, rotated_vectors)[2])
    print(procrustes(vectors, torch.randn(100, 3))[2])