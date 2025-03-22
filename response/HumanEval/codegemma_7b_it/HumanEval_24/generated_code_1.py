from math import sqrt

def method(n):
    """
    Finds the largest number that divides n evenly, smaller than n.

    Args:
        n: The number to find the largest divisor for.

    Returns:
        The largest divisor of n, smaller than n.
    """

    # Initialize the largest divisor to 1.
    largest_divisor = 1

    # Iterate over all the numbers from 2 to the square root of n.
    for i in range(2, int(sqrt(n)) + 1):
        # Check if i divides n evenly.
        if n % i == 0:
            # If i divides n evenly, update the largest divisor.
            largest_divisor = i

    # Return the largest divisor.
    return largest_divisor


# Test case.
n = 12
print(method(n))