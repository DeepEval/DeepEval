import numpy as np

def method(numbers, threshold):
    """
    Check if any two numbers in a list are closer to each other than a given threshold.

    Args:
        numbers (list): A list of numbers to check.
        threshold (float): The minimum distance between two numbers.

    Returns:
        bool: True if any two numbers are closer than the threshold, False otherwise.
    """
    # If the list contains less than two numbers, return False
    if len(numbers) < 2:
        return False

    # Sort the list of numbers
    numbers = sorted(numbers)

    # Iterate over the sorted list to check for any two numbers that are closer than the threshold
    for i in range(len(numbers) - 1):
        # If the difference between two adjacent numbers is less than the threshold, return True
        if numbers[i + 1] - numbers[i] < threshold:
            return True

    # If no two numbers are closer than the threshold, return False
    return False


# Test case
numbers = [10, 15, 20, 25]
threshold = 5
print(method(numbers, threshold))  # Output: True

numbers = [10, 20, 30, 40]
threshold = 10
print(method(numbers, threshold))  # Output: False