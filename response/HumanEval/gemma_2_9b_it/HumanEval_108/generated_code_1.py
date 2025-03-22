def method():
    nums = [12, -3, 4, 0, -12, 56, -78]
    output = count_nums(nums)
    return output

def count_nums(nums):
  """
  Counts the number of integers in an array where the sum of digits is greater than 0.

  Args:
      nums: A list of integers.

  Returns:
      The number of integers in the list whose sum of digits is greater than 0.
  """
  count = 0
  for num in nums:
    sum_of_digits = 0
    num_str = str(abs(num))  # Get the absolute value to handle negative numbers
    for digit in num_str:
      sum_of_digits += int(digit)
    if num < 0:  # Adjust for negative numbers
      sum_of_digits *= -1 
    if sum_of_digits > 0:
      count += 1
  return count

# Test Case
if __name__ == "__main__":
  result = method()
  print(f"Number of integers with positive digit sum: {result}")