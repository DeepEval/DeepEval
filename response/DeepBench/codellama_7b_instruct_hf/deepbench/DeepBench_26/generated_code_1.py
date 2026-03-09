import torch
from torch.nn import functional as F

def _get_perspective_coeffs(startpoints, endpoints):
    # Validate input
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError("Invalid input: startpoints and endpoints must each have four corners.")

    # Get homogeneous coordinates
    startpoints_h = torch.cat((startpoints, torch.ones(4, 1)), dim=1)
    endpoints_h = torch.cat((endpoints, torch.ones(4, 1)), dim=1)

    # Compute perspective matrix
    perspective_matrix = torch.zeros(3, 3)
    perspective_matrix[0, 0] = (startpoints_h[1][0] - startpoints_h[0][0])
    perspective_matrix[0, 1] = (startpoints_h[3][0] - startpoints_h[0][0])
    perspective_matrix[0, 2] = startpoints_h[0][0]
    perspective_matrix[1, 0] = (startpoints_h[2][1] - startpoints_h[0][1])
    perspective_matrix[1, 1] = (startpoints_h[3][1] - startpoints_h[0][1])
    perspective_matrix[1, 2] = startpoints_h[0][1]
    perspective_matrix[2, 0] = (startpoints_h[1][0] - startpoints_h[2][0])
    perspective_matrix[2, 1] = (startpoints_h[3][0] - startpoints_h[2][0])
    perspective_matrix[2, 2] = startpoints_h[2][0]
    perspective_matrix[3, 0] = (startpoints_h[2][1] - startpoints_h[1][1])
    perspective_matrix[3, 1] = (startpoints_h[3][1] - startpoints_h[1][1])
    perspective_matrix[3, 2] = startpoints_h[1][1]

    # Compute perspective coefficients
    a, b, c, d, e, f, g, h = torch.zeros(8)
    a = perspective_matrix[0, 0]
    b = perspective_matrix[0, 1]
    c = perspective_matrix[0, 2]
    d = perspective_matrix[1, 0]
    e = perspective_matrix[1, 1]
    f = perspective_matrix[1, 2]
    g = perspective_matrix[2, 0]
    h = perspective_matrix[2, 1]

    # Return perspective coefficients
    return a, b, c, d, e, f, g, h

if __name__ == "__main__":
    # Create sample input values
    startpoints = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]])
    endpoints = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]])

    # Call function and print results
    a, b, c, d, e, f, g, h = _get_perspective_coeffs(startpoints, endpoints)
    print(a, b, c, d, e, f, g, h)