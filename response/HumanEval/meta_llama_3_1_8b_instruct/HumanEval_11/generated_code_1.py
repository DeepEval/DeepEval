def method():
    """
    This function performs binary XOR operation on two input strings consisting only of 1s and 0s.
    
    Parameters:
    None
    
    Returns:
    str: The result of the binary XOR operation as a string.
    """
    # Take two binary strings as input
    # a = input("Enter the first binary string: ")
    # b = input("Enter the second binary string: ")

    # 修改：
    a = "1010"
    b = "1100" 
    
    # Check if both inputs consist only of 1s and 0s
    if set(a).issubset({'0', '1'}) and set(b).issubset({'0', '1'}):
        # Convert the strings to integers, perform XOR operation and convert back to string
        output = bin(int(a, 2) ^ int(b, 2))[2:]
        
        # Remove leading zeros if present
        output = output.lstrip('0') or '0'
        
        return output
    else:
        return "Error: Input strings must consist only of 1s and 0s."


# Test case
print(method())