import torch

def _get_perspective_coeffs(startpoints, endpoints):
    if len(startpoints) != len(endpoints) or len(startpoints) != 4:
        raise ValueError("Exactly four corners must be provided for both inputs.")

    # Convert to tensors
    startpoints = torch.tensor(startpoints, dtype=torch.float32)
    endpoints = torch.tensor(endpoints, dtype=torch.float32)

    # Define the matrix A
    A = torch.zeros((8, 8), dtype=torch.float32)
    A[:4, :4] = endpoints
    A[:4, 4:] = endpoints
    A[4:, :4] = endpoints.t()
    A[4:, 4:] = startpoints

    # Compute the coefficients using least squares
    coefficients = torch.linalg.lstsq(A, torch.zeros((8,), dtype=torch.float32))[0]

    # Convert to single precision
    coefficients = coefficients.float()

    return coefficients

if __name__ == "__main__":
    # Sample input values
    startpoints = [[10, 10], [100, 10], [10, 100], [100, 100]]
    endpoints = [[20, 20], [120, 20], [20, 120], [120, 120]]

    # Call the function and print the results
    coefficients = _get_perspective_coeffs(startpoints, endpoints)
    print("Perspective coefficients:", coefficients)