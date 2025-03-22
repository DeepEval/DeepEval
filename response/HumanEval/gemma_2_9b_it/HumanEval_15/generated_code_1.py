def method(n):
  """
  Returns a string containing space-delimited numbers starting from 0 upto n inclusive.

  Args:
      n: An integer representing the upper limit of the range.

  Returns:
      A string containing space-delimited numbers from 0 to n.
  """
  output = ' '.join(map(str, range(n + 1)))
  return output

# Test Case
n = 5
result = method(n)
print(f"Numbers from 0 to {n}: {result}")