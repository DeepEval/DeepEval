def method(input_string):
    # Initialize the sum to zero
    sum_ascii_upper = 0
    
    # Iterate over each character in the input string
    for char in input_string:
        # Check if the character is an uppercase letter
        if char.isupper():
            # Add the ASCII value of the uppercase character to the sum
            sum_ascii_upper += ord(char)
    
    # Return the calculated sum
    return sum_ascii_upper

# Test case
test_string = "Hello World! ABCdef"
# Expected output: ASCII sum of 'H', 'W', 'A', 'B', 'C' = 72 + 87 + 65 + 66 + 67 = 357
print(method(test_string))  # Output should be 357