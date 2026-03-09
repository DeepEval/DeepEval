import torch
import torch.linalg
import warnings

def safe_solve_with_mask(A, B):
    if not isinstance(B, torch.Tensor):
        raise AssertionError("B must be a tensor")
    
    if B.dtype not in [torch.float32, torch.float64]:
        B = B.cast(torch.float32)
    
    if torch.__version__ < '1.10':
        warnings.warn("PyTorch version is less than 1.10. Using _torch_solve_cast solution method.")
        A_LU, valid_mask = None, None
        if B.shape == A.shape:
            X = torch.solve(B, A, out=None)
        else:
            X = torch.matmul(torch.inverse(A), B)
        return X, A_LU, valid_mask
    else:
        A_LU = torch.lu(A)
        if B.shape == A.shape:
            X = torch.lu_solve(B, A_LU)
        else:
            X = torch.lu_solve(B.unsqueeze(0), A_LU.unsqueeze(0)).squeeze(0)
        valid_mask = torch.all(torch.linalg.matrix_rank(A) > 0)
        valid_mask = torch.ones_like(valid_mask).bool()
        return X, A_LU, valid_mask

if __name__ == "__main__":
    torch.manual_seed(0)
    B = torch.randn(3, 3, dtype=torch.float32)
    A = torch.randn(3, 3, dtype=torch.float32)
    
    X, A_LU, valid_mask = safe_solve_with_mask(A, B)
    
    print("Solution X:")
    print(X)
    
    print("\nLU Decomposition A_LU:")
    print(A_LU)
    
    print("\nValid Mask:")
    print(valid_mask)