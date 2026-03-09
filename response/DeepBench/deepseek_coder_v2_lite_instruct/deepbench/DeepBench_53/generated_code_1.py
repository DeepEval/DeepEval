import numpy as np

def _coefficients_no_filter(f, degree, use_broadcasting=False):
    if isinstance(degree, tuple):
        d0, d1 = degree
        degree = max(d0, d1)
    else:
        degree = abs(degree)
    
    n = 2 * degree + 1
    k = np.arange(-degree, degree + 1)
    if use_broadcasting:
        x = np.arange(2 * np.pi / (2 * degree))
        k = k[:, np.newaxis]
        x = x[np.newaxis, :]
        coefficients = np.mean(f(x) * np.exp(-1j * k * x), axis=1)
    else:
        coefficients = np.zeros(n, dtype=complex)
        for i, ki in enumerate(k):
            coefficients[i] = np.mean(f(2 * np.pi * ki / n) * np.exp(-1j * ki * 2 * np.pi / n))
    
    return coefficients

if __name__ == "__main__":
    # Example usage
    def sample_function(x):
        return np.sin(x) + 0.5 * np.sin(2 * x)

    coefficients = _coefficients_no_filter(sample_function, degree=2)
    print(coefficients)