def method(xs, x):
  """
  Evaluates a polynomial with coefficients xs at point x.

  Args:
    xs: A list of coefficients of the polynomial, in descending order of power.
    x: The point at which to evaluate the polynomial.

  Returns:
    The value of the polynomial at x.
  """
  output = 0
  for i, coef in enumerate(xs):
    output += coef * (x ** (len(xs) - i - 1))
  return output

# Test case
coefficients = [1, 2, 3]  # Represents the polynomial x^2 + 2x + 3
point = 2
result = method(coefficients, point)
print(f"The value of the polynomial at x = {point} is: {result}")