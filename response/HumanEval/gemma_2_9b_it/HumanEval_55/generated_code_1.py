def method(n):
  """
  Returns the n-th Fibonacci number.

  Args:
      n: An integer representing the desired Fibonacci number.

  Returns:
      An integer representing the n-th Fibonacci number.
  """
  if n <= 1:
    return n
  else:
    return method(n-1) + method(n-2)

# Test case
n = 10
output = method(n)
print(f"The {n}-th Fibonacci number is: {output}")