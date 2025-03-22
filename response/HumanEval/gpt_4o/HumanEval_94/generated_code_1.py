import math

def is_prime(n):
    """Helper function to check if a number is prime."""
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

def sum_of_digits(n):
    """Helper function to calculate the sum of digits of a number."""
    return sum(int(digit) for digit in str(n))

def method(numbers):
    # Filter out the prime numbers from the list
    prime_numbers = [num for num in numbers if is_prime(num)]
    
    # If there are no prime numbers, handle this situation
    if not prime_numbers:
        return 0
    
    # Find the largest prime number
    largest_prime = max(prime_numbers)
    
    # Calculate the sum of its digits
    output = sum_of_digits(largest_prime)
    
    return output

# Test case for validation
def test_method():
    numbers = [10, 3, 5, 7, 12, 13, 17, 19, 23, 29]
    result = method(numbers)
    print(f"Largest prime digit sum in {numbers} is: {result}")
    assert result == 11, "Test case failed!"  # 29 is the largest prime and its digit sum is 2 + 9 = 11

test_method()