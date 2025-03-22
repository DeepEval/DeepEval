def method(xs):
  """
  Calculates the derivative of a polynomial.

  Args:
    xs: A list of coefficients representing the polynomial. 
        xs[0] + xs[1] * x + xs[2] * x^2 + ...

  Returns:
    A list of coefficients representing the derivative of the polynomial.
  """
  derivative_coefficients = []
  for i in range(1, len(xs)):
    derivative_coefficients.append(i * xs[i])
  return derivative_coefficients

# Test case
coefficients = [1, 2, 3, 4]  # Represents x^3 + 2x^2 + 3x + 1
derivative_coefficients = method(coefficients)
print(f"Original polynomial: {coefficients}")
print(f"Derivative: {derivative_coefficients}")