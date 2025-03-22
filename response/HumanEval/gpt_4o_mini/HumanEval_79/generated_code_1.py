def method(decimal_number):
    # Convert decimal to binary using bin() and remove the '0b' prefix
    binary_string = bin(decimal_number)[2:]  # Get binary string without '0b'
    
    # Add 'db' at the beginning and end
    formatted_string = 'db' + binary_string + 'db'
    
    return formatted_string

# Test case for validation
if __name__ == "__main__":
    decimal_number = 10  # Example decimal number
    output = method(decimal_number)
    print(output)  # Should print 'db1010db'