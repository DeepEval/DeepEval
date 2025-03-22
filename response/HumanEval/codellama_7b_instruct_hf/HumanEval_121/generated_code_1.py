from typing import List

def method(nums: List[int]) -> int:
    """
    Given a non-empty list of integers, return the sum of all of the odd elements that are in even positions.

    Args:
        nums: A non-empty list of integers.

    Returns:
        The sum of all of the odd elements that are in even positions.
    """

    # Iterate over the list of integers.
    odd_sum = 0
    for i in range(0, len(nums), 2):
        # Check if the element at the current index is odd.
        if nums[i] % 2 != 0:
            # Add the element to the sum.
            odd_sum += nums[i]

    # Return the sum.
    return odd_sum


# Test case.
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(method(nums))