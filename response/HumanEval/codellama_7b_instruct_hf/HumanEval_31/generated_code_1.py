import math

def method(number):
    """
    This function returns true if a given number is prime, and false otherwise.

    Args:
        number: The number to be checked.

    Returns:
        True if the number is prime, false otherwise.
    """

    if number < 2:
        return False

    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False

    return True


# Test case
number = 13
print(method(number))