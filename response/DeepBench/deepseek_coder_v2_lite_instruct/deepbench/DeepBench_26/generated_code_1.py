import torch

def _get_perspective_coeffs(startpoints, endpoints):
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError("Exactly four corners are required for both inputs.")

    # Create matrices for the system of equations
    X = torch.zeros((8, 8), dtype=torch.double)
    Y = torch.zeros((8, 1), dtype=torch.double)

    for i in range(4):
        x, y = startpoints[i]
        X[2 * i, :] = [x, y, 1, 0, 0, 0, -x * endpoints[i][0], -y * endpoints[i][0]]
        X[2 * i + 1, :] = [0, 0, 0, x, y, 1, -x * endpoints[i][1], -y * endpoints[i][1]]
        Y[2 * i] = endpoints[i][0]
        Y[2 * i + 1] = endpoints[i][1]

    # Solve the system using least squares
    coeffs = torch.linalg.lstsq(X, Y).solution

    # Convert to single precision
    coeffs = coeffs.type(torch.float32)

    return coeffs

if __name__ == "__main__":
    # Sample input values
    startpoints = [[0, 0], [1, 0], [0, 1], [1, 1]]
    endpoints = [[0, 0], [1, 0], [0, 1], [1, 1]]

    # Call the function and print the results
    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    print(coeffs)