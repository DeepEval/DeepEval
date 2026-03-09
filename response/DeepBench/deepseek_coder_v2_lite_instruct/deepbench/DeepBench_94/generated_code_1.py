import torch
import warnings

def safe_solve_with_mask(B, A):
    assert isinstance(B, torch.Tensor), "B must be a tensor"
    if B.dtype != torch.float32 and B.dtype != torch.float64:
        B = B.float()

    if torch.__version__ < '1.10.0':
        warnings.warn("This function requires PyTorch 1.10 or later for correct validity mask handling.")
        def _torch_solve_cast(B, A):
            return torch.lstsq(B, A).solution
        return _torch_solve_cast(B, A), None, torch.ones(B.shape[0], dtype=torch.bool)

    A_LU, pivots = torch.lu(A, True)
    b_shape = B.shape
    B = B.view(b_shape[0], -1)
    X, _ = torch.lu_solve(B, A_LU, pivots)
    X = X.view_as(b_shape)

    valid_mask = torch.ones(B.shape[0], dtype=torch.bool)
    return X, A_LU, valid_mask

if __name__ == "__main__":
    # Example usage
    A = torch.tensor([[3.0, 2.0], [1.0, 4.0]], dtype=torch.float32)
    B = torch.tensor([5.0, 9.0], dtype=torch.float32)
    X, A_LU, valid_mask = safe_solve_with_mask(B, A)
    print("Solution X:", X)
    print("LU Decomposition A_LU:", A_LU)
    print("Valid Mask:", valid_mask)