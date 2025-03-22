def method(numbers):
  """
  Inserts a delimiter between every two consecutive elements of a list.

  Args:
    numbers: A list of numbers.

  Returns:
    A string with delimiters between the numbers.
  """
  output = ','.join(str(num) for num in numbers)
  return output

# Test case
numbers = [1, 2, 3, 4, 5]
result = method(numbers)
print(f"Input: {numbers}")
print(f"Output: {result}")