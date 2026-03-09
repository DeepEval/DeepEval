import torch
import math

def change_basis_real_to_complex(k, dtype=None, device=None):
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device('cpu')

    Q = torch.zeros((2 * k + 1, 2 * k + 1), dtype=dtype, device=device)

    for m in range(-k, k + 1):
        if m < 0:
            Q[k + m, k + m] = 1 / math.sqrt(2)
            Q[k + m, k - m] = -1j / math.sqrt(2)
        elif m == 0:
            Q[k + m, k + m] = 1
        else:
            Q[k + m, k + m] = 1 / math.sqrt(2)
            Q[k + m, k - m] = 1j / math.sqrt(2)
    
    return Q

if __name__ == "__main__":
    k = 2
    dtype = torch.complex64
    device = torch.device('cpu')

    Q = change_basis_real_to_complex(k, dtype, device)
    print(Q)