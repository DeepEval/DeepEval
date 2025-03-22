def method(decimal_number):
    # Convert the decimal number to binary format without the '0b' prefix
    binary_representation = bin(decimal_number)[2:]
    
    # Format the binary representation with 'db' at the beginning and end
    output = f"db{binary_representation}db"
    
    return output

# Test case for validation
test_number = 10  # Binary representation is '1010'
print(method(test_number))  # Expected output: 'db1010db'