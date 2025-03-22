def method(a, b):
    # Set to store unique even digits
    even_digits = set()
    
    # Iterate through each number from a to b
    for num in range(a, b + 1):
        # Convert the number to string to access each digit
        for digit in str(num):
            if digit.isdigit():  # Check if the character is a digit
                digit_value = int(digit)
                if digit_value % 2 == 0:  # Check if the digit is even
                    even_digits.add(digit_value)  # Add it to the set
    
    # Sort the even digits and convert back to a list
    output = sorted(even_digits)
    return output

# Test case
a = 10
b = 25
output = method(a, b)
print(output)  # Expected output: [0, 2, 4, 6, 8]