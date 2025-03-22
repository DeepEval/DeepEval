from collections import defaultdict

def pairs_sum_to_zero(nums):
    """
    This function takes a list of integers as an input.
    It returns True if there are two distinct elements in the list that sum to zero, and False otherwise.
    """

    seen = defaultdict(int)
    for num in nums:
        complement = -num
        if complement in seen and seen[complement] != num:
            return True
        seen[num] += 1

    return False


# Test case
nums = [2, -2, 0, 1, -1]
output = pairs_sum_to_zero(nums)
print(output)  # Output: True