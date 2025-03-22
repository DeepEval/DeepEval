def is_prime(n):
    """Check if a number is a prime number."""
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

def method(n, x, y):
    """Return x if n is prime, otherwise return y."""
    if is_prime(n):
        return x
    else:
        return y

# Test case
if __name__ == "__main__":
    # Test case 1: n is a prime number
    n = 7
    x = "Prime"
    y = "Not Prime"
    result = method(n, x, y)
    print(f"Test Case 1 - Input: n={n}, x={x}, y={y}; Output: {result}")  # Output should be "Prime"
    
    # Test case 2: n is not a prime number
    n = 10
    result = method(n, x, y)
    print(f"Test Case 2 - Input: n={n}, x={x}, y={y}; Output: {result}")  # Output should be "Not Prime"