import math

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

def method(interval1, interval2):
    """Check if the length of the intersection of two intervals is prime."""
    start1, end1 = interval1
    start2, end2 = interval2
    
    # Calculate the length of the intersection
    length = max(0, min(end1, end2) - max(start1, start2))
    
    # Check if the length of the intersection is a prime number
    return is_prime(length)

# Test case
test_interval1 = (1, 2)
test_interval2 = (2, 3)
print(method(test_interval1, test_interval2))  # Expected output: True