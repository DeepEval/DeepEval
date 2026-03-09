import torch

def so3_generators(k: int) -> torch.Tensor:
    # Calculate the dimension of the representation
    dim = 2 * k + 1

    # Initialize generators
    Jx = torch.zeros((dim, dim), dtype=torch.float32)
    Jy = torch.zeros((dim, dim), dtype=torch.float32)
    Jz = torch.zeros((dim, dim), dtype=torch.float32)

    for m in range(dim):
        # Calculate the m value in the representation
        m_val = k - m

        if m < dim - 1:
            # Off-diagonal elements for Jx
            Jx[m, m + 1] = Jx[m + 1, m] = 0.5 * torch.sqrt((k - m_val) * (k + m_val + 1))

            # Off-diagonal elements for Jy
            Jy[m, m + 1] = -0.5j * torch.sqrt((k - m_val) * (k + m_val + 1))
            Jy[m + 1, m] = 0.5j * torch.sqrt((k - m_val) * (k + m_val + 1))

        # Diagonal elements for Jz
        Jz[m, m] = m_val

    return torch.stack([Jx.real, Jz, Jy.real])

if __name__ == "__main__":
    # Example: Get the SO(3) generators for k=1
    k = 1
    generators = so3_generators(k)
    print("Jx:")
    print(generators[0])
    print("\nJz:")
    print(generators[1])
    print("\nJy:")
    print(generators[2])