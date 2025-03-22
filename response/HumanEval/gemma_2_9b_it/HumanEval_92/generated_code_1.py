def method(num1, num2, num3):
  """
  Checks if one number is equal to the sum of the other two.

  Args:
      num1: The first integer.
      num2: The second integer.
      num3: The third integer.

  Returns:
      True if one number is equal to the sum of the other two, and all numbers are integers. 
      False otherwise.
  """
  if isinstance(num1, int) and isinstance(num2, int) and isinstance(num3, int):
    return (num1 == num2 + num3) or (num2 == num1 + num3) or (num3 == num1 + num2)
  else:
    return False

# Test Case
output = method(1, 2, 3) 
print(f"Output: {output}")