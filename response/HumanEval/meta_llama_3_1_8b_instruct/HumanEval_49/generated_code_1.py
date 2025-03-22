import math

def method(n, p):
    """
    Returns 2^n modulo p.
    
    Args:
    n (int): The power to which 2 should be raised.
    p (int): The number by which the result should be moduloed.
    
    Returns:
    int: 2^n modulo p.
    """
    # Use the pow function with three arguments to compute (2^n) % p
    # This is more efficient and accurate than computing 2^n and then taking the modulo
    return pow(2, n, p)

# Test case
n = 10
p = 10007
output = method(n, p)
print(f"2^{n} % {p} = {output}")