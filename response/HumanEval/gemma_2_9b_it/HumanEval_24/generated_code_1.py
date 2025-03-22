def method(n):
    """
    Finds the largest number that divides n evenly, smaller than n.

    Args:
        n: The input number.

    Returns:
        The largest divisor of n smaller than n.
    """
    for i in range(n // 2, 0, -1):  
        if n % i == 0:
            return i  
    return 1  

# Test case
n = 12
output = method(n)
print(f"The largest divisor of {n} smaller than {n} is: {output}")