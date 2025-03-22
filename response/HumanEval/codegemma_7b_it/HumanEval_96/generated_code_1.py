import math

def method(n):
    """
    Returns an array of the first n integers that are prime numbers and less than n.
    """
    primes = []
    for i in range(2, n):
        is_prime = True
        for j in range(2, int(math.sqrt(i)) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    return primes


# Test case
n = 10
output = method(n)
print(output)  # Output: [2, 3, 5, 7]