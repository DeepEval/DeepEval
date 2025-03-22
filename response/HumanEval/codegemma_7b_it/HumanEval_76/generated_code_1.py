from math import log

def is_simple_power(x):
    """
    Checks if a number x is a simple power of n.
    x is a simple power of n if n**int=x
    """
    if x <= 0:
        return False
    
    n = int(log(x, 2))
    return 2**n == x


# Test case
x = 256
print(is_simple_power(x))  # Output: True