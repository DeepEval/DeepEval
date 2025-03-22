def is_prime(n):
    """Check if a number is prime."""
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

def method(number):
    """Return true if the given number is prime, and false otherwise."""
    output = is_prime(number)
    return output

# Test case
test_number = 29  # Example test case, 29 is a prime number
output = method(test_number)
print(f"The number {test_number} is prime: {output}")