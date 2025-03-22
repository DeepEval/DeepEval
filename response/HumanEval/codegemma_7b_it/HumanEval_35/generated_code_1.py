from typing import List

def method(nums: List[int]) -> int:
    """
    This function returns the maximum element in the list.
    """
    return max(nums)

# Test case
nums = [1, 2, 3, 4, 5]
print(method(nums))  # Output: 5