def method(number):
    # Check if the number is positive
    if number < 0:
        raise ValueError("Input must be a positive floating point number.")
    
    # Get the integer part
    integer_part = int(number)
    
    # Calculate the decimal part
    decimal_part = number - integer_part
    
    return decimal_part

# Test case for validation
if __name__ == "__main__":
    test_number = 5.67
    output = method(test_number)
    print(f"The decimal part of {test_number} is {output:.2f}")  # Expected output: 0.67