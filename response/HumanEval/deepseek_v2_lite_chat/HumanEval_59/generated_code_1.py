import math

def method(n):
    # Function to check if a number is prime
    def is_prime(num):
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

    # Find all factors of n
    factors = []
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            factors.append(n // i)
    
    # Find the largest prime factor
    largest_prime = None
    for factor in factors:
        if is_prime(factor):
            if largest_prime is None or factor > largest_prime:
                largest_prime = factor
    
    return largest_prime

# Example test case
# n = 100
# Should return 5, as the largest prime factor of 100 is 5