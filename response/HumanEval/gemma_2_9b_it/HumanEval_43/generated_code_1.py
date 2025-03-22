def method():
    nums = [1, -1, 2, -2, 3, -3, 4, -4]
    output = pairs_sum_to_zero(nums)
    return output

def pairs_sum_to_zero(nums):
    """
    Checks if there are two distinct elements in a list that sum to zero.

    Args:
        nums: A list of integers.

    Returns:
        True if there are two distinct elements that sum to zero, False otherwise.
    """
    seen = set()
    for num in nums:
        complement = -num
        if complement in seen:
            return True
        seen.add(num)
    return False

# Test case
print(method())  # Output: True