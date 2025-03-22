def is_prime(n):
    """Helper function to check if a number is prime."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def get_primes_below(limit):
    """Returns a list of prime numbers below the given limit."""
    primes = []
    for num in range(2, limit):
        if is_prime(num):
            primes.append(num)
    return primes

def method(num):
    primes = get_primes_below(100)
    
    # Check all combinations of three primes
    for i in range(len(primes)):
        for j in range(i, len(primes)):
            for k in range(j, len(primes)):
                product = primes[i] * primes[j] * primes[k]
                if product == num:
                    return True
    return False

# Test case
output = method(30)  # 30 = 2 * 3 * 5, which are all prime numbers
print(output)  # Expected output: True

output = method(28)  # 28 is not the product of three primes
print(output)  # Expected output: False