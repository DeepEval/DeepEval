def method(coefficients, x):
    """
    Evaluates a polynomial with coefficients xs at point x.
    Returns the polynomial evaluated at x.

    Args:
    coefficients (list of float): List of coefficients of the polynomial, from 0 to n.
    x (float): The point at which to evaluate the polynomial.

    Returns:
    float: The value of the polynomial at x.
    """
    # Initialize the result to the constant term of the polynomial
    result = coefficients[0]

    # Evaluate the polynomial starting from the first term
    for i in range(1, len(coefficients)):
        result += coefficients[i] * (x ** i)

    return result

# Example usage:
# coefficients = [1, 2, 3, 4]
# x = 0.5
# output = 1 + 2*0.5 + 3*(0.5)^2 + 4*(0.5)^3
# print(method(coefficients, x)) # Should print the output