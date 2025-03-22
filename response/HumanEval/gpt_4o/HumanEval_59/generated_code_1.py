import math

def largest_prime_factor(n):
    # This function will find the largest prime factor of n
    def is_prime(num):
        """Check if a number is prime."""
        if num <= 1:
            return False
        if num <= 3:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True

    largest_factor = -1
    
    # Check the number of 2s that divide n
    while n % 2 == 0:
        largest_factor = 2
        n = n // 2
    
    # n must be odd at this point, hence a skip of 2 (i = i + 2) can be used
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        # While i divides n, save i and divide n
        while n % i == 0:
            largest_factor = i
            n = n // i

    # This condition is to check if n is a prime number greater than 2
    if n > 2:
        largest_factor = n

    return largest_factor

def method(n):
    output = largest_prime_factor(n)
    return output

# Test case to validate the function
print(method(13195))  # Expected output: 29