from functools import partial
from itertools import product

import numpy as np

def _coefficients_no_filter(f, degree, use_broadcasting=True):
    """Computes the first 2d+1 Fourier coefficients for a 2pi-periodic function.

    Args:
        f: Callable representing the function to be Fourier-transformed.
        degree: Integer or tuple representing the maximum frequency.
        use_broadcasting: Boolean deciding whether to use broadcasting for simultaneous function evaluations.

    Returns:
        NumPy array of complex numbers containing the Fourier coefficients up to the specified degree.
    """

    dim = len(f(next(iter(product(range(degree))))))

    if isinstance(degree, int):
        degree = (degree,) * dim

    if use_broadcasting:
        x = np.asarray(list(product(range(degree))))
        y = np.broadcast_to(x, (2,) + x.shape)
        vals = np.array(f(y.tolist()))
        vals = vals.reshape((vals.shape[0], -1))
        values = np.fft.fft(vals)
        values = values / (2 ** dim)
        return values[:, :degree[0] + 1]

    values = np.zeros((degree[0] + 1, dim), dtype=complex)
    for freq in product(range(degree[0] + 1)):
        x = np.asarray(list(product(range(degree))))
        x = np.dot(x, np.array(freq).T) * (2 * np.pi / (2 ** dim))
        y = f(x)
        values[freq, :] = y.mean(axis=0)

    return values

if __name__ == "__main__":
    # Sample input values
    f = lambda x: np.sin(x)
    degree = (5,)

    # Call the function and print the results
    coefficients = _coefficients_no_filter(f, degree, use_broadcasting=True)
    print(coefficients)