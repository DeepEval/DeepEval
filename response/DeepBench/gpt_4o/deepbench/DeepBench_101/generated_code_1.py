import torch
import numpy as np

def su2_generators(k):
    j = k / 2
    dim = int(2 * j + 1)
    J_plus = np.zeros((dim, dim), dtype=complex)
    J_minus = np.zeros((dim, dim), dtype=complex)
    J_z = np.zeros((dim, dim), dtype=complex)

    for m in range(dim - 1):
        m_val = j - m
        J_plus[m, m + 1] = np.sqrt((j - m_val) * (j + m_val + 1))
        J_minus[m + 1, m] = np.sqrt((j + m_val) * (j - m_val + 1))
        J_z[m, m] = m_val

    J_x = 0.5 * (J_plus + J_minus)
    J_y = -0.5j * (J_plus - J_minus)

    J_x = torch.tensor(J_x, dtype=torch.complex64)
    J_y = torch.tensor(J_y, dtype=torch.complex64)
    J_z = torch.tensor(J_z, dtype=torch.complex64)

    return torch.stack([J_x, J_y, J_z])

if __name__ == "__main__":
    k = 1  # Example representation index
    generators = su2_generators(k)
    print("J_x:\n", generators[0])
    print("J_y:\n", generators[1])
    print("J_z:\n", generators[2])