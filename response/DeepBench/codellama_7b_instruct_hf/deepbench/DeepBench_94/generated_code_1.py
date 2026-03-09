import torch
import numpy as np
from torch.linalg import lu_solve

def safe_solve_with_mask(A, B):
    if torch.__version__ < "1.10":
        return _torch_solve_cast(A, B)
    else:
        # Perform LU decomposition of A
        A_LU = lu_solve(A, torch.eye(A.shape[0], device=A.device))

        # Compute the solution X using LU solve
        X = lu_solve(A_LU, B)

        # Compute the validity mask
        valid_mask = torch.all(torch.abs(A_LU) > 1e-5, dim=1)

        return X, A_LU, valid_mask

# Minimal runnable example
if __name__ == "__main__":
    # Create sample input values
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[5], [6]], dtype=torch.float32)

    # Call the function
    X, A_LU, valid_mask = safe_solve_with_mask(A, B)

    # Print the results
    print(X)
    print(A_LU)
    print(valid_mask)