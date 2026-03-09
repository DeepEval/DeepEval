import torch
import warnings

def safe_solve_with_mask(B, A):
    assert isinstance(B, torch.Tensor), "B must be a torch.Tensor"
    
    if B.dtype not in [torch.float32, torch.float64]:
        B = B.float()
    
    if A.dtype not in [torch.float32, torch.float64]:
        A = A.float()
    
    # Check PyTorch version
    pytorch_version = torch.__version__.split('.')
    major_version = int(pytorch_version[0])
    minor_version = int(pytorch_version[1])
    
    if major_version < 1 or (major_version == 1 and minor_version < 10):
        warnings.warn("PyTorch version is less than 1.10. Validity mask might not be correct.")
        
        def _torch_solve_cast(B, A):
            return torch.solve(B, A)
        
        X, _ = _torch_solve_cast(B, A)
        # LU decomposition is not available in versions < 1.10
        A_LU = None
        valid_mask = torch.ones(B.size(0), dtype=torch.bool)
    else:
        # Use LU decomposition and solve
        try:
            A_LU, pivots = torch.lu(A, get_infos=False)
            X = torch.lu_solve(B, A_LU, pivots)
            valid_mask = torch.ones(B.size(0), dtype=torch.bool)
        except RuntimeError as e:
            # Handle singular matrix case
            X = torch.zeros_like(B)
            A_LU = A.clone()  # LU decomposition not successful
            valid_mask = torch.zeros(B.size(0), dtype=torch.bool)
            warnings.warn(f"Singular matrix encountered: {e}")
    
    return X, A_LU, valid_mask

if __name__ == "__main__":
    # Example inputs
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    B = torch.tensor([[9.0], [8.0]], dtype=torch.float32)
    
    X, A_LU, valid_mask = safe_solve_with_mask(B, A)
    
    print("Solution X:")
    print(X)
    print("LU Decomposition of A:")
    print(A_LU)
    print("Validity Mask:")
    print(valid_mask)