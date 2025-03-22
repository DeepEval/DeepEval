def method(hex_string):
    # Define the set of prime hexadecimal digits
    prime_hex_digits = {'2', '3', '5', '7', 'B', 'D'}

    # Count the number of prime hexadecimal digits in the input string
    count = sum(1 for char in hex_string if char in prime_hex_digits)

    return count

# Test case to validate the function
test_hex_string = "1234ABCD56789"
output = method(test_hex_string)
print(f"The number of prime hexadecimal digits in '{test_hex_string}' is: {output}")