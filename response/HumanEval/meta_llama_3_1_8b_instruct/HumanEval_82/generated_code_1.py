import math

def is_prime(n):
    """
    Checks if a number is prime.

    Args:
    n (int): The number to check.

    Returns:
    bool: True if the number is prime, False otherwise.
    """
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    sqrt_n = math.isqrt(n)
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def method(s):
    """
    Checks if the length of a string is a prime number.

    Args:
    s (str): The input string.

    Returns:
    bool: True if the length of the string is a prime number, False otherwise.
    """
    return is_prime(len(s))

# Test case
input_string = "Hello"
output = method(input_string)
print(f"Input string: {input_string}")
print(f"Is the length of the string a prime number? {output}")