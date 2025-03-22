def method():
    # Example input
    number = 3.75
    
    # Calculate the integer part
    integer_part = int(number)
    
    # Calculate the decimal part
    decimal_part = number - integer_part
    
    # Return the decimal part
    return decimal_part

# Test case
output = method()
print(f"The decimal part of 3.75 is {output}")  # Expected output: 0.75