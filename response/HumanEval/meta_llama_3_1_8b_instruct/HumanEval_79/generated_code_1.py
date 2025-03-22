def method(n):
    """
    Convert a decimal number to binary format with extra characters 'db' at the beginning and end.

    Args:
        n (int): The decimal number to be converted.

    Returns:
        str: The binary string with extra characters 'db' at the beginning and end.
    """
    # Convert the decimal number to binary and remove the '0b' prefix
    binary = bin(n)[2:]
    
    # Add the extra characters 'db' at the beginning and end of the string
    output = 'db' + binary + 'db'
    
    return output

# Test case
print(method(10))  # Output: db1010db