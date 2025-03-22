import math

def method(n):
    """
    Return the largest prime factor of n. Assume n > 1 and is not a prime.
    """
    # Find the largest prime factor of n
    prime_factor = n
    for i in range(2, int(math.sqrt(n)) + 1):
        while n % i == 0:
            prime_factor = i
            n //= i
    return prime_factor

# Test case for validation
n = 12
output = method(n)
print(output) # Output: 3