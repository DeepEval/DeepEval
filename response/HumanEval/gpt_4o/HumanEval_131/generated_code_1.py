def method(n):
    # Convert the number to a string to iterate over each character
    n_str = str(n)
    
    # Initialize the product of odd digits
    product = 1
    has_odd_digit = False
    
    # Iterate over each digit
    for digit in n_str:
        digit_int = int(digit)
        # Check if the digit is odd
        if digit_int % 2 != 0:
            product *= digit_int
            has_odd_digit = True
    
    # If no odd digits were found, return 0
    if not has_odd_digit:
        return 0
    
    return product

# Test case
n = 123456
output = method(n)
print(output)  # Expected output is 3 * 5 = 15