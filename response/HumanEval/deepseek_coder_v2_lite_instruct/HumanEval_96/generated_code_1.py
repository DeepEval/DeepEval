def is_prime(num):
    """Check if a number is a prime number."""
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

def method(n):
    """Return an array of the first n integers that are prime numbers and less than n."""
    primes = []
    for i in range(2, n):
        if is_prime(i):
            primes.append(i)
    return primes

# Test case
output = method(30)
print(output)  # Expected output: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]