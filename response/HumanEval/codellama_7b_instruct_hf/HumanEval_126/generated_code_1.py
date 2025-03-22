from collections import Counter

def method(nums):
  """
  Checks if a list of numbers is sorted in ascending order and has no duplicate elements.

  Args:
    nums: A list of integers.

  Returns:
    True if the list is sorted and has no duplicates, False otherwise.
  """

  # Check for duplicates
  if len(nums) != len(set(nums)):
    return False

  # Check if the list is sorted
  for i in range(1, len(nums)):
    if nums[i] < nums[i - 1]:
      return False

  return True

# Test case
nums = [1, 2, 3, 4]
print(method(nums)) # Output: True

nums = [1, 2, 2, 4]
print(method(nums)) # Output: False

nums = [1, 2, 3, 3]
print(method(nums)) # Output: False