from collections import Counter

def method(nums):
    """
    Given a non-empty list of positive integers, returns the greatest integer greater than zero with a frequency greater than or equal to its value.

    Args:
        nums: A non-empty list of positive integers.

    Returns:
        The greatest integer meeting the requirement, or -1 if no such integer exists.
    """

    # Count the frequency of each integer in the list.
    frequencies = Counter(nums)

    # Find the greatest integer with a frequency greater than or equal to its value.
    for num, freq in frequencies.items():
        if freq >= num:
            return num

    # No such integer exists.
    return -1

# Test case.
nums = [2, 2, 4, 4, 5, 5, 5]
output = method(nums)
print(output)