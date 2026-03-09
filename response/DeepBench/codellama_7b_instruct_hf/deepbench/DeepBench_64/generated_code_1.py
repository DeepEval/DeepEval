import torch

def rinverse(M, sym=False):
    """Matrix inversion of rightmost dimensions (batched). For 1, 2, and 3 dimensions this uses the formulae. For larger matrices, it uses blockwise inversion to reduce to smaller matrices."""
    if M.ndimension() == 1:
        return M.inverse()
    elif M.ndimension() == 2:
        return torch.inverse(M)
    elif M.ndimension() == 3:
        return torch.inverse(M)
    else:
        # Use blockwise inversion to reduce to smaller matrices
        B = M.shape[-1]
        b = M.shape[-2]
        blocks = M.split(B, dim=-1)
        out = torch.empty(b, B, B, dtype=M.dtype, device=M.device)
        for i in range(b):
            out[i] = rinverse(blocks[i], sym=sym)
        return out

if __name__ == "__main__":
    # Test the function with a sample input
    M = torch.randn(10, 10, 10)
    out = rinverse(M)
    print(out)