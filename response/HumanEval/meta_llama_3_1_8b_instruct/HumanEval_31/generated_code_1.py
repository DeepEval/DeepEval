def method(n):
    """
    Returns True if a given number is prime, and False otherwise.

    Args:
        n (int): The number to check for primality.

    Returns:
        bool: True if the number is prime, False otherwise.
    """
    if n <= 1:  # 1 and numbers less than 1 are not prime
        return False
    if n == 2:  # 2 is the only even prime number
        return True
    if n % 2 == 0:  # all other even numbers are not prime
        return False

    # Check odd factors up to the square root of n
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False

    return True


# Test cases
print(method(25))  # False
print(method(23))  # True
print(method(37))  # True
print(method(48))  # False
print(method(2))   # True
print(method(1))   # False