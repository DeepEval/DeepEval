def is_prime(num):
    """Check if a number is a prime number."""
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def method(n):
    """Return an array of the first n prime numbers that are less than n."""
    primes = []
    current = 2  # Start checking for prime numbers from 2
    
    while len(primes) < n:
        if current >= n:
            break  # Stop if current exceeds n
        if current < n and is_prime(current):
            primes.append(current)
        current += 1
    
    return primes

# Example test case
output = method(5)  # We want the first 5 prime numbers less than 5
print(output)  # Expected output: [2, 3]