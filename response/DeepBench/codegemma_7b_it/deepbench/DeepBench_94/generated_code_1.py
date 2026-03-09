import torch

def safe_solve_with_mask(B, A):
    # Input handling
    assert isinstance(B, torch.Tensor), "B must be a tensor"
    B = B.to(torch.float32)  # Ensure B is float32 for compatibility

    # PyTorch version compatibility
    if torch.__version__ < "1.10":
        A_LU, _ = torch.lu_unpack(torch.lu(A))
        X = torch._torch_solve_cast(B, A)
        valid_mask = torch.ones_like(X[:, 0], dtype=torch.bool)
        print("Warning: PyTorch version is less than 1.10. Validity mask may not be correct.")
    else:
        A_LU, piv = torch.lu_factor(A)
        X, piv_inv = torch.lu_solve((A_LU, piv), B)
        valid_mask = piv_inv.gt(0)

    return X, A_LU, valid_mask

if __name__ == "__main__":
    # Sample input data
    B = torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 0]])
    A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

    # Call the function and print results
    X, A_LU, valid_mask = safe_solve_with_mask(B, A)
    print("Solution:", X)
    print("LU decomposition:", A_LU)
    print("Valid mask:", valid_mask)