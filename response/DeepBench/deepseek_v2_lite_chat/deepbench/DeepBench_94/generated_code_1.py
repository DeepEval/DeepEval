import torch
import torch.linalg as la

def safe_solve_with_mask(B, A, dtype=torch.float32):
    if not isinstance(B, torch.Tensor):
        raise AssertionError("Right-hand side tensor B must be provided.")
    
    if B.dtype not in [torch.float32, torch.float64]:
        B = B.to(dtype=torch.float32)
    
    # Ensure A is a tensor
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
    
    # LU decomposition
    A_LU, _ = la.lu(A, pivot=True)
    
    # Solve AX = B
    try:
        X = la.solve(A, B, some=True)
    except la.LinAlgError:
        # Singular matrix detected
        # Fall back to a different method for backward compatibility
        X = _torch_solve_cast(A, B)
        print("Warning: Singular matrix encountered, using fallback method.")
        mask = torch.ones_like(B, dtype=torch.bool)
    else:
        mask = None
    
    return X, A_LU, mask

def _torch_solve_cast(A, B):
    # Simple back-compat function that casts B to float32 and solves AX = B
    return la.solve(A, B).type(torch.float32)

if __name__ == "__main__":
    # Example usage
    A = torch.tensor([[2, 1], [1, 2]], dtype=torch.float32)
    B = torch.tensor([[3], [3]], dtype=torch.float32)
    X, _, mask = safe_solve_with_mask(B, A)
    print("Solution X:", X)
    print("Valid solutions mask:", mask)

    # Check if the mask indicates the correct solutions
    valid_mask = mask.bool()
    X_true = torch.tensor([[1.5], [1.5]], dtype=torch.float32)
    print("Expected solution X (true):", X_true)
    print("Valid solutions mask (expected):", valid_mask)