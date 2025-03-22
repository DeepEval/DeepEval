def method(x, base):
    """
    Convert a number `x` from base 10 to a specified base `base`.
    
    Args:
    x (int): The number to be converted.
    base (int): The base to convert `x` to.
    
    Returns:
    str: The string representation of `x` in the specified base.
    """
    convertString = "0123456789"
    if x < base:
        return convertString[x]
    else:
        return method(x // base, base) + convertString[x % base]

# Test case
print(method(123, 2))  # Output: 1111011
print(method(10, 8))   # Output: 12
print(method(255, 16)) # Output: ff