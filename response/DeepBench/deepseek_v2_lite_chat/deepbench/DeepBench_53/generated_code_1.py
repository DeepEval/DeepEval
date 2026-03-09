import numpy as np

def _coefficients_no_filter(f, degree, use_broadcasting):
    x = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    y = f(x)

    # Compute Fourier coefficients
    if use_broadcasting:
        coefficients = np.fft.fft(y)
    else:
        diff = x[1] - x[0]
        coefficients = np.zeros(degree + 1, dtype=np.complex)
        for n in range(degree + 1):
            coefficients[n] = np.trapz(y * np.cos(n * x), x) - 0.5j * np.trapz(y * np.sin(n * x), x)

    return coefficients

if __name__ == "__main__":
    # Define a simple function to serve as f
    def f(x):
        return np.sin(x)  # A sine wave of frequency 1

    # Calculate Fourier coefficients for degree 2
    degree = 2
    coefficients = _coefficients_no_filter(f, degree, use_broadcasting=True)
    print("Fourier coefficients for degree", degree, ":", coefficients)