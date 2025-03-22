def method(decimal_number):
    # Convert the decimal number to binary
    binary_string = bin(decimal_number)[2:]
    
    # Add 'db' at the beginning and end of the binary string
    output = 'db' + binary_string + 'db'
    
    return output

# Test case
decimal_number = 13
print(method(decimal_number))  # Expected output: 'db1101db'