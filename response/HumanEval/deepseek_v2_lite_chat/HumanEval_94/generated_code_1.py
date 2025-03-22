from typing import List, Union

def is_prime(n: int) -> bool:
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

def method():
    # List of integers to check for prime numbers
    integers = [10, 15, 21, 35, 40, 60, 78, 111, 133, 141, 143]
    
    # Initialize variables to find the largest prime and its digit sum
    largest_prime = None
    digit_sum = 0
    
    # Iterate through the list of integers
    for num in integers:
        # Check if the number is prime
        if is_prime(num):
            # Update largest prime and digit sum if necessary
            if largest_prime is None or num > largest_prime:
                largest_prime = num
                digit_sum = sum(int(digit) for digit in str(num))
    
    # Return the largest prime number and its digit sum
    return largest_prime, digit_sum

# Test case
# assert method() == (143, 8)  # Example test case
print(method())