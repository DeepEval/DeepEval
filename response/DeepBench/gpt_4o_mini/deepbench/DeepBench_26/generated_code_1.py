import torch

def _get_perspective_coeffs(startpoints, endpoints):
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError("Both startpoints and endpoints must contain exactly four coordinate pairs.")

    A = torch.zeros((8, 8), dtype=torch.float64)
    B = torch.zeros((8, 1), dtype=torch.float64)

    for i in range(4):
        x, y = startpoints[i]
        u, v = endpoints[i]
        A[i * 2] = torch.tensor([x, y, 1, 0, 0, 0, -u * x, -u * y], dtype=torch.float64)
        A[i * 2 + 1] = torch.tensor([0, 0, 0, x, y, 1, -v * x, -v * y], dtype=torch.float64)
        B[i * 2] = u
        B[i * 2 + 1] = v

    coeffs, _ = torch.lstsq(B, A)
    coeffs = coeffs[:8].reshape(-1).float()  # Convert to single precision

    return coeffs

if __name__ == "__main__":
    startpoints = [(0, 0), (1, 0), (1, 1), (0, 1)]
    endpoints = [(0, 0), (2, 0), (2, 2), (0, 2)]
    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    print(coeffs)