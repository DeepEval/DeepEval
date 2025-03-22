def method():
    nums = [1, 2, 3, 3, 4, 5]
    output = is_sorted_ascending(nums)
    return output

def is_sorted_ascending(nums):
    """
    Checks if a list of numbers is sorted in ascending order and has at most one duplicate of each number.

    Args:
      nums: A list of integers.

    Returns:
      True if the list is sorted in ascending order and has at most one duplicate of each number, False otherwise.
    """
    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]:
            return False

        # Check for more than one duplicate
        if i > 0 and nums[i] == nums[i - 1] and nums[i] == nums[i + 1]:
            return False
    return True

# Test case
print(method()) # Output: True