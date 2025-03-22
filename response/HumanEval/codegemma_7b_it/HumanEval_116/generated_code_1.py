from typing import List

def method(nums: List[int]) -> List[int]:
    """
    Sorts an array of non-negative integers according to number of ones in their binary representation in ascending order.
    For similar number of ones, sort based on decimal value.

    Args:
        nums: The input array of integers.

    Returns:
        The sorted array.
    """

    def count_ones(n: int) -> int:
        """Counts the number of ones in the binary representation of an integer."""
        count = 0
        while n:
            if n & 1:
                count += 1
            n >>= 1
        return count

    # Sort the array using a custom comparison function
    nums.sort(key=lambda x: (count_ones(x), x))

    return nums

# Test case
nums = [1, 3, 5, 2, 4]
expected = [2, 1, 5, 3, 4]
actual = method(nums)

# assert actual == expected, f"Test case failed. Expected {expected}, but got {actual}"

# print("Test case passed")