def method():
    n = 600851475143
    largest_prime_factor = 1

    # Divide out all factors of 2
    while n % 2 == 0:
        largest_prime_factor = 2
        n //= 2

    # Check for odd factors from 3 onwards
    factor = 3
    while factor * factor <= n:
        while n % factor == 0:
            largest_prime_factor = factor
            n //= factor
        factor += 2

    # If n is a prime number greater than 2
    if n > 2:
        largest_prime_factor = n

    return largest_prime_factor

# Test case
output = method()
print("The largest prime factor is:", output)