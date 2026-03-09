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
    
    determinants = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return determinants

if __name__ == "__main__":
    sample_input = torch.tensor([[[1, 2, 3], [0, 1, 4], [5, 6, 0]],
                                  [[1, 0, 2], [3, 4, 5], [6, 7, 8]]], dtype=torch.float32)
    result = _safe_det_3x3(sample_input)
    print(result)