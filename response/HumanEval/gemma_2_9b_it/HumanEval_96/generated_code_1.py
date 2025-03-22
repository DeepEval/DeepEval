def method(n):
    """
    Finds the first n prime numbers less than n.

    Args:
        n: A non-negative integer.

    Returns:
        A list of the first n prime numbers less than n.
    """

    primes = []
    num = 2

    while len(primes) < n:
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1

    return primes

# Test case
n = 10
output = method(n)
print(f"The first {n} prime numbers are: {output}")