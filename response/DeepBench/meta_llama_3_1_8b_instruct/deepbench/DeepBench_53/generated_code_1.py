import numpy as np

def _coefficients_no_filter(f, degree, use_broadcasting):
    if isinstance(degree, tuple):
        degree = degree[0]

    N = degree + 1
    k = np.arange(N)
    k = k.reshape((1, N)) if use_broadcasting else k

    coefficients = np.zeros((1, N), dtype=np.complex128)
    coefficients[:, :degree+1] = (np.cos(k * np.pi) + 1j * np.sin(k * np.pi) * 2) / N * f(np.pi * k)

    if use_broadcasting:
        coefficients = coefficients.squeeze()

    return coefficients

if __name__ == "__main__":
    f = lambda x: np.sin(x) + 2 * np.cos(x)
    degree = 3
    use_broadcasting = True

    coefficients = _coefficients_no_filter(f, degree, use_broadcasting)
    print(coefficients)