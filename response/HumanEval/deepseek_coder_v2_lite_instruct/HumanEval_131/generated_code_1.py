def method(n):
    # Convert the integer to a string to iterate over its digits
    str_n = str(n)
    
    # Initialize the product of odd digits
    product = 1
    
    # Flag to check if there are any odd digits
    has_odd_digit = False
    
    # Iterate over each digit in the string
    for digit in str_n:
        # Convert the digit back to an integer
        num = int(digit)
        
        # Check if the digit is odd
        if num % 2 != 0:
            product *= num
            has_odd_digit = True
    
    # Return 0 if all digits are even
    if not has_odd_digit:
        return 0
    
    return product

# Test case
output = method(123456)
print(output)  # Expected output: 45