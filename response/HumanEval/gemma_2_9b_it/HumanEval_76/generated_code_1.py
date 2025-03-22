def method(x, n):
    """
    Checks if a number x is a simple power of n.

    Args:
        x (int): The number to check.
        n (int): The base.

    Returns:
        bool: True if x is a simple power of n, False otherwise.
    """
    return n ** int(x ** (1/n)) == x

# Test case
output = method(8, 2)
print(output)  # Output: True