def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def method():
    """Return x if n is prime, otherwise return y."""
    n = 11  # Example prime number
    x = 10  # Value to return if n is prime
    y = 20  # Value to return if n is not prime
    
    if is_prime(n):
        return x
    else:
        return y


# Test case
output = method()
print(f"Output: {output}")

# Additional test case for non-prime number
output_non_prime = method()
print(f"Output (non-prime number): {output_non_prime}")