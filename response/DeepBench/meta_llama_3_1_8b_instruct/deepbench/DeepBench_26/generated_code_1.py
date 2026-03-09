import torch
import numpy as np
from typing import List, Tuple

def _get_perspective_coeffs(startpoints: List[Tuple[float, float]], endpoints: List[Tuple[float, float]]) -> torch.Tensor:
    if len(startpoints)!= 4 or len(endpoints)!= 4:
        raise ValueError("Exactly four corners must be provided for both inputs.")

    # Convert inputs to double precision to ensure numerical stability
    startpoints = [tuple(float(x) for x in point) for point in startpoints]
    endpoints = [tuple(float(x) for x in point) for point in endpoints]

    # Create matrices for least squares calculation
    A = np.zeros((8, 9))
    b = np.zeros(8)

    for i in range(4):
        x1, y1 = startpoints[i]
        x2, y2 = endpoints[i]
        x3, y3 = startpoints[(i + 1) % 4]
        x4, y4 = endpoints[(i + 1) % 4]

        A[2 * i] = [x1, y1, 1, 0, 0, 0, -x1 * x3, -y1 * x3, -x1 * x3 * x3]
        A[2 * i + 1] = [0, 0, 0, x1, y1, 1, -x1 * y3, -y1 * y3, -x1 * y3 * y3]

        b[2 * i] = x2 * x3
        b[2 * i + 1] = y2 * y3

    A = torch.DoubleTensor(A)
    b = torch.DoubleTensor(b)

    # Solve for coefficients using least squares
    coeffs = torch.linalg.lstsq(A, b, rcond=None)[0]

    # Convert results to single precision before returning
    return coeffs.float()

if __name__ == "__main__":
    # Create sample input values
    startpoints = [(0, 0), (100, 0), (100, 100), (0, 100)]
    endpoints = [(10, 10), (110, 10), (110, 110), (10, 110)]

    # Call the function and print the results
    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    print(coeffs)