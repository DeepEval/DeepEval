import torch

def change_basis_real_to_complex(k, dtype=None, device=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')

    Q = torch.zeros((k**2, k**2), dtype=dtype, device=device)
    idx = 0

    for ell in range(k):
        for m in range(-ell, ell + 1):
            for n in range(k**2):
                ell_n = n // (2 * ell + 1)
                m_n = n % (2 * ell + 1) - ell
                if m_n == m:
                    Q[n, idx] = (-1)**m
                    Q[n, idx + 1] = 0
                else:
                    Q[n, idx] = 0
                    Q[n, idx + 1] = (-1)**m
            idx += 2

    return Q

if __name__ == "__main__":
    k = 3
    Q = change_basis_real_to_complex(k)
    print(Q)