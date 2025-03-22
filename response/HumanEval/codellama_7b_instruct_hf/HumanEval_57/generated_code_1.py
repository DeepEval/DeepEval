from typing import List

def is_monotonic(nums: List[int]) -> bool:
    """
    Returns True if list elements are monotonically increasing or decreasing.

    Args:
    nums: A list of integers.

    Returns:
    True if list elements are monotonically increasing or decreasing, False otherwise.
    """

    # Check for empty list or single element list
    if not nums or len(nums) == 1:
        return True

    # Determine the direction of monotonicity
    direction = nums[1] - nums[0]

    # Check if all elements maintain the same direction
    for i in range(2, len(nums)):
        if (nums[i] - nums[i - 1]) * direction < 0:
            return False

    return True


# Test case
nums = [1, 2, 3, 4, 5]
print(is_monotonic(nums)) # Output: True

nums = [5, 4, 3, 2, 1]
print(is_monotonic(nums)) # Output: True

nums = [1, 2, 2, 2, 3]
print(is_monotonic(nums)) # Output: True

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]
print(is_monotonic(nums)) # Output: False