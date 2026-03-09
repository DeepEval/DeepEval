import numpy as np

def _coefficients_no_filter(f, degree, use_broadcasting=True):
    if isinstance(degree, int):
        degree = (degree, )
    d = degree[0]
    
    n_points = 1000
    x = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    if use_broadcasting:
        f_values = f(x)
    else:
        f_values = np.array([f(xi) for xi in x])
    
    coefficients = []
    for k in range(-d, d + 1):
        exponent = np.exp(-1j * k * x)
        coefficient = (1 / n_points) * np.sum(f_values * exponent)
        coefficients.append(coefficient)
    
    return np.array(coefficients, dtype=complex)

if __name__ == "__main__":
    def sample_function(x):
        return np.sin(x) + 0.5 * np.cos(3 * x)
    
    degree = 5
    coefficients = _coefficients_no_filter(sample_function, degree, use_broadcasting=True)
    print("Fourier coefficients:", coefficients)