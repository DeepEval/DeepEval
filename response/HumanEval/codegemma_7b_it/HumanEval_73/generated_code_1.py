from collections import Counter

def method(arr):
    """
    Given an array arr of integers, find the minimum number of elements that need to be changed to make the array palindromic.

    Args:
        arr (list): An array of integers.

    Returns:
        int: The minimum number of elements that need to be changed.
    """

    # Find the middle element of the array.
    mid = len(arr) // 2

    # Count the number of elements that are not in their correct position.
    changes = 0
    for i in range(mid):
        if arr[i] != arr[-i - 1]:
            changes += 1

    # Return the number of changes.
    return changes


# Test case.
arr = [1, 2, 3, 2, 1]
print(method(arr))  # Output: 3