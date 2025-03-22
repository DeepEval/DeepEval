def method(a, b):
    """
    Function to find the greatest common divisor of two integers a and b.

    Args:
    a (int): The first integer.
    b (int): The second integer.

    Returns:
    int: The greatest common divisor of a and b.
    """
    # Base case: if a is 0, the GCD is b, otherwise, GCD is the same as the GCD of b and a%b
    if a == 0:
        return b
    else:
        return method(b % a, a)

# Test case
def test_method():
    """
    Test function to validate the 'method()' function.
    """
    # assert method(48, 18) == 6
    # assert method(101, 100) == 1
    # assert method(1000, 100) == 100
    # assert method(1024, 192) == 64
    # print("All test cases pass")
    print(method(48,18))

# Run the test function
test_method()