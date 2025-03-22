from sympy import isprime, primerange
from itertools import combinations

def is_product_of_three_primes(n):
    # Function to check if a number is the product of three primes
    def is_product_of_primes(num, primes):
        for prime in primes:
            if num % prime == 0 and is_prime(num // prime):
                return True
        return False

    # Function to find all prime numbers less than the given number
    def get_primes(limit):
        return [p for p in primerange(2, limit + 1) if isprime(p)]

    # Function to check if a number is prime
    def is_prime(num):
        return num > 1 and all(num % i for i in range(2, int(num**0.5) + 1))

    # Get all prime numbers less than n
    primes = get_primes(n)

    # Find all combinations of primes that multiply to n
    prime_combinations = set()
    for comb in combinations(primes, 3):
        if is_product_of_primes(n, comb):
            prime_combinations.add(tuple(sorted(comb)))

    # Check if there are exactly three combinations
    return len(prime_combinations) == 3

# Test case
def test_function():
    # assert is_product_of_three_primes(30) == False  # 30 is 2 * 3 * 5
    # assert is_product_of_three_primes(18) == True   # 18 is 2 * 3 * 3
    # assert is_product_of_three_primes(3) == True    # 3 is 3 * 1
    # assert is_product_of_three_primes(50) == False  # 50 is not the product of exactly three primes
    print(is_product_of_three_primes(50))

test_function()