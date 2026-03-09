import numpy as np

def coefficients_no_filter(f, degree, use_broadcasting=True):
    # Compute the first (2d + 1) Fourier coefficients for a 2pi-periodic function
    # without filtering.
    #
    # Parameters
    # ----------
    # f : callable
    #     The function to compute the Fourier coefficients for.
    # degree : int or tuple of ints
    #     The maximum degree of the Fourier coefficients to compute.
    # use_broadcasting : bool, optional
    #     Whether to use broadcasting for simultaneous function evaluations.
    #     Default is True.
    #
    # Returns
    # -------
    # coeffs : numpy.ndarray
    #     The Fourier coefficients up to the specified degree.

    # Check the input arguments
    if not callable(f):
        raise ValueError("f must be a callable function.")
    if not isinstance(degree, (int, tuple)):
        raise ValueError("degree must be an integer or tuple of integers.")
    if not isinstance(use_broadcasting, bool):
        raise ValueError("use_broadcasting must be a boolean value.")

    # Get the input dimensions
    input_shape = np.shape(f(np.zeros((2,))))
    num_dims = len(input_shape)

    # Initialize the Fourier coefficients array
    coeffs = np.zeros((num_dims, degree + 1), dtype=complex)

    # Compute the Fourier coefficients
    for k in range(degree + 1):
        # Compute the Fourier series
        if use_broadcasting:
            # Use broadcasting for simultaneous function evaluations
            f_vals = np.empty((*input_shape, k + 1), dtype=complex)
            for i in range(k + 1):
                f_vals[..., i] = f(np.arange(i + 1, dtype=float) / (k + 1))
        else:
            # Use a for-loop for function evaluations
            f_vals = np.empty(k + 1, dtype=complex)
            for i in range(k + 1):
                f_vals[i] = f(np.arange(i + 1, dtype=float) / (k + 1))

        # Compute the Fourier coefficients
        coeffs[..., k] = f_vals / (k + 1)

    # Return the Fourier coefficients
    return coeffs

if __name__ == "__main__":
    # Create a sample input function
    def f(x):
        return np.sin(x)

    # Compute the Fourier coefficients
    coeffs = coefficients_no_filter(f, 5)

    # Print the Fourier coefficients
    print(coeffs)