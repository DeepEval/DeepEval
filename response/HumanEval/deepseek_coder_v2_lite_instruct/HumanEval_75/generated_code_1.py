from itertools import combinations

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

def generate_primes(limit):
    primes = []
    for num in range(2, limit):
        if is_prime(num):
            primes.append(num)
    return primes

def method():
    given_number = 84  # Example test case
    primes = generate_primes(100)
    for combo in combinations(primes, 3):
        if combo[0] * combo[1] * combo[2] == given_number:
            return True
    return False

# Test case
output = method()
print(output)  # Expected output: True if 84 is the product of three prime numbers, otherwise False