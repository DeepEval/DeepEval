def method(n):
    # Initialize the product to 1 and a flag to check for odd digits
    product = 1
    has_odd_digit = False

    # Convert the integer to a string to iterate over each digit
    for digit in str(n):
        digit = int(digit)  # Convert the digit back to an integer
        if digit % 2 != 0:  # Check if the digit is odd
            product *= digit  # Multiply to the product
            has_odd_digit = True  # Set flag to True if we found an odd digit

    # If no odd digits were found, return 0
    if not has_odd_digit:
        return 0

    return product  # Return the final product of odd digits


# Test case
n = 123456789
output = method(n)
print(f"The product of the odd digits in {n} is: {output}")  # Expected output: 945 (1 * 3 * 5 * 7 * 9)