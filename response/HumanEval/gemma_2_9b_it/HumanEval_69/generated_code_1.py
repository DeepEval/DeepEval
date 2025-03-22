def method(nums):
    """
    Finds the greatest integer in a list that has a frequency greater than or equal to its value.

    Args:
        nums: A list of positive integers.

    Returns:
        The greatest integer meeting the criteria, or -1 if none exists.
    """
    counts = {}
    for num in nums:
        if num in counts:
            counts[num] += 1
        else:
            counts[num] = 1

    max_integer = -1
    for num in nums:
        if counts[num] >= num and num > max_integer:
            max_integer = num

    return max_integer

# Test Case
nums = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
output = method(nums)
print(f"Input: {nums}")
print(f"Output: {output}")