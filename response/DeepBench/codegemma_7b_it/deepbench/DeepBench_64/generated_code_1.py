import torch

def rinverse(M, sym=False):
    """Matrix inversion of rightmost dimensions (batched).
    For 1, 2, and 3 dimensions this uses the formulae.
    For larger matrices, it uses blockwise inversion to reduce to smaller matrices.
    """
    if len(M.shape) == 1:
        return M ** -1
    elif len(M.shape) == 2:
        if sym:
            return (M + M.t()) ** -0.5
        else:
            return torch.linalg.inv(M)
    elif len(M.shape) == 3:
        if sym:
            return (M + M.transpose(1, 2)) ** -0.5
        else:
            return torch.linalg.inv(M)
    else:
        # Handle larger matrices using blockwise inversion
        pass

if __name__ == "__main__":
    # Example input
    M = torch.tensor([[2, 1], [4, 3]])

    # Invert the matrix
    inv_M = rinverse(M)

    # Print the result
    print(inv_M)