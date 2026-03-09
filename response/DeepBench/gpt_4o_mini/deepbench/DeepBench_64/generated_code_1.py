import torch

def rinverse(M, sym=False):
    if sym:
        M = (M + M.transpose(-1, -2)) / 2  # Ensure symmetry if required

    if M.dim() == 1:
        return 1 / M  # Inverse for 1D (vector)
    elif M.dim() == 2:
        return torch.linalg.inv(M)  # Inverse for 2D (matrix)
    elif M.dim() == 3:
        return torch.linalg.inv(M)  # Inverse for 3D (batch of matrices)
    else:
        # For higher dimensions, we will blockwise invert
        shape = M.shape
        num_batches = shape[:-2]  # All dimensions except the last two
        M_flat = M.view(-1, shape[-2], shape[-1])  # Flatten to 2D
        M_inv_flat = torch.linalg.inv(M_flat)  # Invert batch
        return M_inv_flat.view(*num_batches, shape[-2], shape[-1])  # Reshape back

if __name__ == "__main__":
    # Example to validate the function
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2D matrix
    B = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # 3D tensor

    inv_A = rinverse(A)
    inv_B = rinverse(B)

    print("Inverse of A:\n", inv_A)
    print("Inverse of B:\n", inv_B)