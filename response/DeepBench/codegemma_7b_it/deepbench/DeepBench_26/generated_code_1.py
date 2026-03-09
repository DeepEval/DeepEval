import torch

def _get_perspective_coeffs(startpoints, endpoints):
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError(f"Expected 4 corner points, but got {len(startpoints)} for startpoints and {len(endpoints)} for endpoints")

    # Convert to float64 for stability
    startpoints = torch.as_tensor(startpoints, dtype=torch.float64)
    endpoints = torch.as_tensor(endpoints, dtype=torch.float64)

    # Form the A and b matrices for the least squares problem
    A = torch.stack([
        [1, startpoints[0, 0], startpoints[0, 1], 1, 0, 0, -startpoints[0, 0] * endpoints[0, 0], -startpoints[0, 1] * endpoints[0, 0]],
        [0, 0, 0, 0, 1, startpoints[0, 0], -startpoints[0, 0] * endpoints[0, 1], -startpoints[0, 1] * endpoints[0, 1]],
        [1, startpoints[1, 0], startpoints[1, 1], 1, 0, 0, -startpoints[1, 0] * endpoints[1, 0], -startpoints[1, 1] * endpoints[1, 0]],
        [0, 0, 0, 0, 1, startpoints[1, 0], -startpoints[1, 0] * endpoints[1, 1], -startpoints[1, 1] * endpoints[1, 1]],
        [1, startpoints[2, 0], startpoints[2, 1], 1, 0, 0, -startpoints[2, 0] * endpoints[2, 0], -startpoints[2, 1] * endpoints[2, 0]],
        [0, 0, 0, 0, 1, startpoints[2, 0], -startpoints[2, 0] * endpoints[2, 1], -startpoints[2, 1] * endpoints[2, 1]],
        [1, startpoints[3, 0], startpoints[3, 1], 1, 0, 0, -startpoints[3, 0] * endpoints[3, 0], -startpoints[3, 1] * endpoints[3, 0]],
        [0, 0, 0, 0, 1, startpoints[3, 0], -startpoints[3, 0] * endpoints[3, 1], -startpoints[3, 1] * endpoints[3, 1]],
    ], dtype=torch.float64)

    b = torch.as_tensor(endpoints.flatten(), dtype=torch.float64)

    # Solve the least squares problem
    x = torch.linalg.lstsq(A, b, rcond=None).solution

    # Convert to single precision and return
    return x.float()

if __name__ == "__main__":
    # Sample input values
    startpoints = [[0, 0], [100, 0], [0, 100], [100, 100]]
    endpoints = [[20, 30], [120, 30], [20, 130], [120, 130]]

    # Call the function and print the results
    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    print(coeffs)