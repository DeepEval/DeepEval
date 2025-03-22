import math

def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def prime_factors(n):
    """Find all prime factors of a number."""
    factors = []
    for i in range(2, n + 1):
        while n % i == 0 and is_prime(i):
            factors.append(i)
            n = n // i
    return factors

def method(n):
    """
    Check if the given number is the multiplication of 3 prime numbers.
    
    Args:
        n (int): The number to check.
    
    Returns:
        bool: True if the number is the multiplication of 3 prime numbers, False otherwise.
    """
    factors = prime_factors(n)
    # Check if the number of prime factors is 3
    if len(factors) == 3:
        return True
    return False

# Test case
output = method(30)  # 30 = 2 * 3 * 5
print("Method:", method(30))

# Test case with a number that is not the product of 3 prime numbers
output = method(12)  # 12 = 2 * 2 * 3
print("Method:", method(12))

# Test case with a number that is less than 100
output = method(40)  # 40 = 2 * 2 * 2 * 5
print("Method:", method(40))