def method():
    import sys
    # Importing the sys module to read from standard input
    
    # Reading the input strings
    a = sys.stdin.readline().strip()
    b = sys.stdin.readline().strip()
    
    # Converting the strings to integers to perform the XOR operation
    a_int = int(a, 2)
    b_int = int(b, 2)
    
    # Performing the XOR operation
    result = bin(a_int ^ b_int)[2:]
    
    # Return the result as a string
    return result