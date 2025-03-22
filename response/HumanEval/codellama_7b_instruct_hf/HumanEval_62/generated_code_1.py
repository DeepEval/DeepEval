import numpy as np

def method(xs):
    """
    Returns the derivative of the polynomial in the same form.

    Args:
        xs: A list of coefficients of the polynomial.

    Returns:
        A list of coefficients of the derivative polynomial.
    """

    # Calculate the derivative of the polynomial.
    der_xs = xs * np.arange(1, len(xs) + 1)

    # Return the derivative polynomial.
    return der_xs

# Test case.
xs = [1, 2, 3]
der_xs = method(xs)

print(der_xs)