import torch
import warnings

def safe_solve_with_mask(B, A):
    assert isinstance(B, torch.Tensor), "B must be a tensor."
    
    if B.dtype not in [torch.float32, torch.float64]:
        B = B.to(torch.float32)

    if torch.__version__ < '1.10':
        warnings.warn("PyTorch version is less than 1.10; falling back to alternative solution method.")
        # Assume _torch_solve_cast is defined elsewhere for older versions
        X, valid_mask = _torch_solve_cast(B, A)
        return X, None, valid_mask

    A_LU, pivots = torch.linalg.lu(A)
    valid_mask = torch.ones(A.shape[0], dtype=torch.bool)
    
    try:
        X = torch.linalg.solve(A_LU, B)
    except RuntimeError as e:
        if 'singular' in str(e):
            valid_mask = torch.zeros(A.shape[0], dtype=torch.bool)
            X = torch.linalg.lstsq(A, B, rcond=None).solution
        else:
            raise

    return X, A_LU, valid_mask

if __name__ == "__main__":
    A = torch.tensor([[2.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    B = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    X, A_LU, valid_mask = safe_solve_with_mask(B, A)
    
    print("Solution X:")
    print(X)
    print("LU Decomposition A_LU:")
    print(A_LU)
    print("Valid Mask:")
    print(valid_mask)