import torch

def inverse(M, sym=False):
    if len(M.shape) == 2:
        # Special case for 2D matrices
        return M.inverse()
    elif len(M.shape) == 3:
        # Batch inversion for 3D matrices
        batch_size = M.shape[0]
        inverted_matrices = []
        for i in range(batch_size):
            M_batch = M[i]
            if sym:
                # Blockwise inversion for symmetric matrices
                M_batch = M_batch.block_diag(*[M_batch.clone() for _ in range(2)])
            inverted_matrices.append(M_batch.inverse())
        return torch.stack(inverted_matrices, dim=0)
    else:
        raise ValueError("The matrix must be a 2D or 3D tensor.")

if __name__ == "__main__":
    # Sample input values
    M = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    
    # Call the function and print the results
    result = inverse(M)
    print("Inverted matrix:\n", result)