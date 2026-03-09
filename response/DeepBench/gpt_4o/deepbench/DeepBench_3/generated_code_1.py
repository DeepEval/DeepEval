import torch

def _safe_det_3x3(t):
    a = t[:, 0, 0]
    b = t[:, 0, 1]
    c = t[:, 0, 2]
    d = t[:, 1, 0]
    e = t[:, 1, 1]
    f = t[:, 1, 2]
    g = t[:, 2, 0]
    h = t[:, 2, 1]
    i = t[:, 2, 2]
    
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    
    return det

if __name__ == "__main__":
    # Create a batch of 3x3 matrices
    t = torch.tensor([
        [[1.0, 2.0, 3.0],
         [0.0, 1.0, 4.0],
         [5.0, 6.0, 0.0]],
        [[2.0, 5.0, 3.0],
         [1.0, 0.0, 8.0],
         [7.0, 6.0, 4.0]]
    ])

    # Calculate determinants
    det = _safe_det_3x3(t)
    print(det)