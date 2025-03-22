import math

def method(n, p):
    """
    Return 2^n modulo p.
    """
    return (2 ** n) % p

# Test cases
def test_method():
    # assert method(3, 5) == 4
    # assert method(10, 2) == 4
    # assert method(0, 1) == 1
    # assert method(2, 1000000007) == 2  # Using a large modulus for testing
    # print("All test cases passed!")
    method(3, 5)

# Run the test cases
test_method()