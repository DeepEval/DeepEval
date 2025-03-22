def method(nums):
    """
    Returns True if there are three distinct elements in the list that sum to zero, and False otherwise.

    Args:
    nums (list): A list of integers.

    Returns:
    bool: Whether there are three distinct elements that sum to zero.
    """
    # Check if the input is a list of integers
    if not isinstance(nums, list) or not all(isinstance(num, int) for num in nums):
        raise TypeError("Input must be a list of integers.")

    # Sort the list
    nums.sort()

    # Iterate over the list
    for i in range(len(nums) - 2):
        # Skip the same result
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # Initialize two pointers
        left, right = i + 1, len(nums) - 1

        # Move the pointers
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                return True

    return False


# Test case
nums = [-1, 0, 1, 2, -1, -4]
print(method(nums))  # Expected output: True