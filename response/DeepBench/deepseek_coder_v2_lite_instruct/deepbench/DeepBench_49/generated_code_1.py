import torch

def matrix_cofactor_tensor(matrix):
    # Check if the input matrix is of the required shape
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Input must be a tensor of shape (*, 3, 3)")
    
    # Compute the cofactor matrix for each 3x3 matrix in the batch
    cofactor_matrix = torch.zeros_like(matrix)
    
    for i in range(3):
        for j in range(3):
            minor_matrix = matrix[..., :i, :j] \
                           .reshape(matrix.shape[:-2] + (-1,)) \
                           .cat(matrix[..., :i, j+1:].reshape(matrix.shape[:-2] + (-1,)), dim=-1) \
                           .cat(matrix[..., i+1:, :j].reshape(matrix.shape[:-2] + (-1,)), dim=-1) \
                           .cat(matrix[..., i+1:, j+1:].reshape(matrix.shape[:-2] + (-1,)), dim=-1)
            cofactor_matrix[..., i, j] = (-1) ** (i + j) * torch.det(minor_matrix)
    
    # Check for singular matrices and raise an exception if any is found
    det_matrix = torch.det(matrix)
    if torch.any(det_matrix == 0):
        raise Exception("All input matrices are singular")
    
    return cofactor_matrix

if __name__ == "__main__":
    # Create a sample input tensor of shape (2, 3, 3)
    matrices = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ])
    
    # Call the function and print the results
    try:
        cofactor_matrices = matrix_cofactor_tensor(matrices)
        print("Cofactor Matrices:\n", cofactor_matrices)
    except Exception as e:
        print(e)