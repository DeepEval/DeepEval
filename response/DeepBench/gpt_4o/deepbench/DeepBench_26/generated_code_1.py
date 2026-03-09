import torch

def _get_perspective_coeffs(startpoints, endpoints):
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError("Exactly four corners are required for both startpoints and endpoints.")
    
    startpoints = torch.tensor(startpoints, dtype=torch.float64)
    endpoints = torch.tensor(endpoints, dtype=torch.float64)

    A = []
    B = []

    for i in range(4):
        x, y = startpoints[i]
        x_prime, y_prime = endpoints[i]
        A.append([x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime])
        A.append([0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime])
        B.append(x_prime)
        B.append(y_prime)

    A = torch.tensor(A, dtype=torch.float64)
    B = torch.tensor(B, dtype=torch.float64)

    # Solve Ax = B for x using least squares
    coeffs, _ = torch.lstsq(B, A)

    # Convert to single precision before returning
    coeffs = coeffs[:8].flatten().to(torch.float32)

    return coeffs

if __name__ == "__main__":
    startpoints = [(0, 0), (1, 0), (1, 1), (0, 1)]
    endpoints = [(0, 0), (2, 0), (2, 2), (0, 2)]

    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    print("Perspective transformation coefficients:", coeffs.tolist())