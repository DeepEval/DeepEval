def method(n):
    """
    Return the largest prime factor of n. 
    Assume n > 1 and is not a prime.
    """
    largest_prime = 1
    i = 2
    while i * i <= n:
        if n % i == 0:
            largest_prime = i
            n //= i
        else:
            i += 1
    if n > 1:
        largest_prime = n
    return largest_prime

# Test case
n = 12
output = method(n)
print(f"The largest prime factor of {n} is {output}")