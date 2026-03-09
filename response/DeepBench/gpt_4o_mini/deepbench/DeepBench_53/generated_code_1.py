import numpy as np

def _coefficients_no_filter(f, degree, use_broadcasting=True):
    if isinstance(degree, int):
        degrees = range(-degree, degree + 1)
    else:
        degrees = range(-degree[0], degree[1] + 1)

    if use_broadcasting:
        x = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        f_values = f(x)
        coefficients = np.array([np.trapz(f_values * np.exp(-1j * k * x), x) / (2 * np.pi) for k in degrees])
    else:
        coefficients = np.zeros(len(degrees), dtype=np.complex_)
        x = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        for i, k in enumerate(degrees):
            coefficients[i] = np.trapz(f(x) * np.exp(-1j * k * x), x) / (2 * np.pi)

    return coefficients

if __name__ == "__main__":
    def sample_function(x):
        return np.sin(x) + 0.5 * np.cos(2 * x)

    degree = 3
    coeffs = _coefficients_no_filter(sample_function, degree)
    print("Fourier coefficients:", coeffs)