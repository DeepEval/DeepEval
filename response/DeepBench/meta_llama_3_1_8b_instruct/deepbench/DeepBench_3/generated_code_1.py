import torch

def _safe_det_3x3(t):
    # Calculate the determinant of a batch of 3x3 matrices
    det = torch.zeros(t.shape[0])
    for i in range(t.shape[0]):
        a, b, c = t[i, 0, :]
        d, e, f = t[i, 1, :]
        g, h, _ = t[i, 2, :]
        
        det[i] = a * (e * h - f * g) - b * (d * h - f * g) + c * (d * e - f * g)
    return det

if __name__ == "__main__":
    # Create a batch of 3x3 matrices
    t = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 10]], 
                      [[11, 12, 13], [14, 15, 16], [17, 18, 19]]])
    
    # Calculate the determinants using the custom function
    det_custom = _safe_det_3x3(t)
    
    # Calculate the determinants using torch.det()
    det_torch = torch.det(t)
    
    # Print the results
    print("Determinants using custom function:", det_custom)
    print("Determinants using torch.det():", det_torch)