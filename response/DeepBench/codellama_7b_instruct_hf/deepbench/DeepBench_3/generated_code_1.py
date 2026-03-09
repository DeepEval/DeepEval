import torch

def safe_det_3x3(t):
    """
    Calculates the determinant of a batch of 3x3 matrices.
    The input is a torch.Tensor t of shape (N, 3, 3) where N is the number of matrices in the batch.
    The function returns a torch.Tensor of shape (N) containing the determinants of the input matrices.
    """
    # Calculate the determinant of the first matrix
    det = t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1])
    for i in range(1, t.shape[0]):
        det += t[i][0] * (t[i][1] * t[i][2] - t[i][2] * t[i][1])
    return det

if __name__ == "__main__":
    # Create sample input values
    t = torch.randn(3, 3)

    # Call the function and print the results
    print(safe_det_3x3(t))