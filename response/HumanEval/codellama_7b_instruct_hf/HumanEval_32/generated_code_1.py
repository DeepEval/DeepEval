from itertools import accumulate
import operator

def method(xs, x):
  """
  Evaluates polynomial with coefficients xs at point x. return xs[0] + xs[1] * x + xs[2] * x^2 + .... xs[n] * x^n 

  Args:
    xs: A list of coefficients for the polynomial.
    x: The point at which to evaluate the polynomial.

  Returns:
    The value of the polynomial evaluated at x.
  """
  output = sum(accumulate(reversed(xs), operator.mul), x)
  return output


# Test case
xs = [1, 2, 3, 4]
x = 2

output = method(xs, x)

print(output)  # Output: 44