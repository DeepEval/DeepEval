import torch

def change_basis_real_to_complex(k, dtype=None, device=None):
    # Construct a transformation matrix Q that converts real spherical harmonics into complex spherical harmonics
    Q = torch.eye(k * (k + 2), dtype=dtype, device=device)
    for l in range(k):
        for m in range(l + 1):
            for p in range(k):
                for q in range(p + 1):
                    Q[l * (l + 1) + m, p * (p + 1) + q] = torch.sqrt((2 * l + 1) / (4 * torch.pi)) * torch.cos(torch.tensor(l, dtype=torch.double) * torch.tensor(m, dtype=torch.double) * torch.tensor(p, dtype=torch.double) * torch.tensor(q, dtype=torch.double))
                    Q[l * (l + 1) + m, p * (p + 1) + q] = torch.sqrt((2 * l + 1) / (4 * torch.pi)) * torch.sin(torch.tensor(l, dtype=torch.double) * torch.tensor(m, dtype=torch.double) * torch.tensor(p, dtype=torch.double) * torch.tensor(q, dtype=torch.double))
    return Q

if __name__ == "__main__":
    # Test the function
    k = 5
    Q = change_basis_real_to_complex(k)
    print(Q)