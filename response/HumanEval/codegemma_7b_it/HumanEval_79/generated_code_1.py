import math

def method():
    # Get decimal number from the user
    # decimal_number = int(input("Enter a decimal number: "))
    decimal_number = 10

    # Initialize binary string
    binary_string = "db"

    # Convert decimal number to binary
    while decimal_number > 0:
        remainder = decimal_number % 2
        binary_string = str(remainder) + binary_string
        decimal_number //= 2

    # Return binary string
    return binary_string


# Test case
test_case = 10
print(method())