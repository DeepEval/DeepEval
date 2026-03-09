import torch
import torch.linalg

def rinverse(M, sym=False):
    # Check if the input is a tensor
    if not isinstance(M, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")

    # Check if the tensor has more than 3 dimensions
    if M.dim() > 3:
        # Perform blockwise inversion
        inv_M = []
        for i in range(M.shape[-1]):
            inv_M.append(rinverse(M[..., i:i+1], sym))
        inv_M = torch.cat(inv_M, dim=-1)
    else:
        # Use torch.linalg.inv for small matrices
        if sym:
            inv_M = torch.linalg.potrf(M) @ torch.linalg.triangular_solve(torch.eye(M.shape[-1]), torch.linalg.triangular_solve(torch.eye(M.shape[-1]), torch.linalg.triangular_solve(torch.eye(M.shape[-1]), torch.linalg.triangular_solve(torch.eye(M.shape[-1]), torch.linalg.potrf(M), upper=False), upper=False), upper=False), upper=False)
        else:
            inv_M = torch.linalg.inv(M)
    
    return inv_M

if __name__ == "__main__":
    # Create a sample 3D tensor
    M = torch.randn(2, 2, 2, 2)
    print("Input Matrix:")
    print(M)
    
    # Call the function
    inv_M = rinverse(M)
    print("\nInverse Matrix:")
    print(inv_M)
    
    # Check if the inverse matrix is correct
    print("\nCheck if the product of the original and inverse matrices is close to the identity matrix:")
    print(torch.allclose(M @ inv_M, torch.eye(2).unsqueeze(0).unsqueeze(0)))