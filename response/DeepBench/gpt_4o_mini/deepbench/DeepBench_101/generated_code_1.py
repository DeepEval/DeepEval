import torch

def su2_generators(k):
    j = k
    dim = 2 * j + 1
    J_x = torch.zeros((dim, dim))
    J_y = torch.zeros((dim, dim))
    J_z = torch.zeros((dim, dim))

    for m in range(-j, j + 1):
        if m < j:
            J_plus = torch.zeros((dim, dim))
            J_plus[m + j, m + 1 + j] = 0.5 * ((j - m) * (j + m + 1)) ** 0.5
            J_x += J_plus
        if m > -j:
            J_minus = torch.zeros((dim, dim))
            J_minus[m + j, m - 1 + j] = 0.5 * ((j + m) * (j - m + 1)) ** 0.5
            J_x -= J_minus

    J_y = 1j * (J_plus - J_minus) / 2
    J_z = torch.diag(torch.tensor([m for m in range(-j, j + 1)]))

    return torch.stack([J_x, J_y, J_z])

if __name__ == "__main__":
    k = 1  # Example representation index
    generators = su2_generators(k)
    print(generators)