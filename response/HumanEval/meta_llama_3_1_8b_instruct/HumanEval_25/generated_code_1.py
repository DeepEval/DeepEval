def method(n):
    """
    Returns a list of prime factors of a given integer in the order from smallest to largest.
    Each of the factors is listed number of times corresponding to how many times it appears in the factorization.
    
    Args:
    n (int): The input number to be factored.
    
    Returns:
    list: A list of prime factors.
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

# Test case
input_num = 13194
output = method(input_num)
print(f"The prime factors of {input_num} are: {output}")