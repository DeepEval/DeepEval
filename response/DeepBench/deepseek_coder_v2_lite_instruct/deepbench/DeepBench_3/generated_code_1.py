import torch

def _safe_det_3x3(t):
    # Extract the individual matrices from the batch
    a = t[:, 0, 0]
    b = t[:, 0, 1]
    c = t[:, 0, 2]
    d = t[:, 1, 0]
    e = t[:, 1, 1]
    f = t[:, 1, 2]
    g = t[:, 2, 0]
    h = t[:, 2, 1]
    i = t[:, 2, 2]
    
    # Calculate the determinant using the formula: det(A) = aei + bfg + cdh - ceg - bdi - afh
    det = a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h
    
    return det

if __name__ == "__main__":
    # Create a batch of 3x3 matrices
    matrices = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[4, 2, 3], [1, 5, 6], [7, 8, 9]]
    ])
    
    # Calculate the determinants
    determinants = _safe_det_3x3(matrices)
    
    # Print the results
    print(determinants)