def method(N):
  """
  Given a positive integer N, return the total sum of its digits in binary. 

  Args:
      N: A positive integer.

  Returns:
      The sum of the digits of N in binary.
  """

  binary = bin(N)[2:]  # Convert to binary and remove "0b" prefix
  sum_of_digits = 0
  for digit in binary:
    sum_of_digits += int(digit)
  return sum_of_digits

# Test case
N = 12
output = method(N)
print(f"The sum of digits of {N} in binary is: {output}")