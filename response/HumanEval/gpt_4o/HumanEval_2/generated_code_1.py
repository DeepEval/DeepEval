def method(number):
    # Get the integer part of the number
    integer_part = int(number)
    
    # Subtract the integer part from the original number to get the decimal part
    decimal_part = number - integer_part
    
    return decimal_part

# Test case for validation
test_number = 123.456
output = method(test_number)
print(f"Decimal part of {test_number} is: {output}")  # Expected output: 0.456