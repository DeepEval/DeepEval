def method(n):
    import itertools

    # Generate all prime numbers less than 100
    def is_prime(num):
        if num <= 1:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    primes = [i for i in range(2, 100) if is_prime(i)]

    # Check if n is the product of exactly three primes
    for combo in itertools.combinations_with_replacement(primes, 3):
        if combo[0] * combo[1] * combo[2] == n:
            return True

    return False

# Test case
print(method(30))  # 2 * 3 * 5 = 30, should return True
print(method(60))  # 3 * 4 * 5 = 60, should return False (4 is not prime)