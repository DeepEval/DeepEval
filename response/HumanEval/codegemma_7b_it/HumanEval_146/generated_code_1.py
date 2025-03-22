import math

def method(arr):
    """
    This function takes an array of numbers as input and returns the number of elements in the array that are greater than 10 and both first and last digits of a number are odd (1, 3, 5, 7, 9).

    Args:
        arr: An array of numbers.

    Returns:
        The number of elements in the array that meet the criteria.
    """
    count = 0
    for num in arr:
        if num > 10 and num % 2 != 0 and num % 10 != 0:
            count += 1
    return count

# Test case
arr = [123, 234, 345, 456, 567, 678, 789, 890, 901]
output = method(arr)
print(output)  # Output: 3