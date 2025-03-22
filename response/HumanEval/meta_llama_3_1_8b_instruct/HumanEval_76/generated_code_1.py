import math

def method(x, n):
    """
    This function checks if a number x is a simple power of n.

    Args:
    x (float): The number to check.
    n (float): The base number.

    Returns:
    bool: True if x is a simple power of n, False otherwise.
    """

    # Check if n is zero, as any number to the power of zero is one
    if n == 0 and x!= 1:
        return False

    # Calculate the logarithm of x with base n
    log_result = math.log(x, n)

    # Check if the result is an integer
    return log_result == int(log_result)


# Test case
x = 16
n = 2
output = method(x, n)
print(f"Is {x} a simple power of {n}? {output}")

# Output: Is 16 a simple power of 2? True