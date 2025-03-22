import math

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def method():
    x = 1
    y = 0
    n = 15  # Example input
    if is_prime(n):
        return x
    else:
        return y

# Test case to validate the function
def test_method():
    # assert method() == 15
    # assert method() == 0
    # assert method() == 17  # Prime number
    # assert method() == 20  # Non-prime number
    method()

# Run the test function
test_method()