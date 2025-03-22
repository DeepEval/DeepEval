def method(n, p):
    """
    Calculates 2^n modulo p, handling potential numeric overflow.

    Args:
        n: The exponent.
        p: The modulus.

    Returns:
        The result of 2^n modulo p.
    """

    result = 1
    base = 2
    while n > 0:
        if n % 2 == 1:
            result = (result * base) % p
        base = (base * base) % p
        n //= 2  
    return result

# Test Case
n = 1000
p = 1000000007 
output = method(n, p)
print(f"2^{n} modulo {p} is: {output}")