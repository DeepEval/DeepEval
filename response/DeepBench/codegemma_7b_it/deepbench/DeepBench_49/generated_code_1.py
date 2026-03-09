import torch

def matrix_cofactor_tensor(matrix):
    """Computes the cofactor matrix for a given tensor of matrices with shape (*, 3, 3)."""
    
    # Check if all input matrices are singular
    if torch.any(torch.det(matrix) == 0):
        raise Exception("All input matrices are singular.")

    # Calculate the cofactor matrix for each input matrix
    cofactor_matrices = []
    for mat in matrix:
        cofactor_matrices.append(torch.linalg.det(mat) * torch.tensor([[
            [mat[1, 1], mat[1, 2]],
            [mat[2, 1], mat[2, 2]]
        ],
        [
            [mat[0, 1], mat[0, 2]],
            [mat[2, 1], mat[2, 2]]
        ],
        [
            [mat[0, 1], mat[0, 2]],
            [mat[1, 1], mat[1, 2]]
        ]]) / 2)

    # Return a tensor containing the cofactor matrices
    return torch.stack(cofactor_matrices)

if __name__ == "__main__":
    # Create sample input values
    matrix = torch.randn(2, 3, 3)

    # Call the function and print the results
    cofactor_matrices = matrix_cofactor_tensor(matrix)
    print(cofactor_matrices)