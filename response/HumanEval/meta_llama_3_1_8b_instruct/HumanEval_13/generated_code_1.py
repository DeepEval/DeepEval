import math

def method():
    """
    Returns the greatest common divisor (GCD) of two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The greatest common divisor of a and b.
    """

    # Define the two integers
    a = 48
    b = 18

    # Use the math.gcd() function to calculate the GCD
    gcd = math.gcd(a, b)

    return gcd

# Test the function
output = method()
print("GCD:", output)

# Additional test case
print("Test case 2:")
print("GCD of 12 and 15:", method())