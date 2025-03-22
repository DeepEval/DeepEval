def method(n, x, y):
    """
    This function returns the value of x if n is a prime number and the value of y otherwise.

    Args:
      n: An integer to be checked for primality.
      x: The value to return if n is prime.
      y: The value to return if n is not prime.

    Returns:
      The value of x if n is prime, otherwise the value of y.
    """
    # Prime number check
    if n <= 1:
        return y
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return y
    return x

# Test Case
n = 17
x = 10
y = 20
output = method(n, x, y)
print(f"Output for n = {n}: {output}")