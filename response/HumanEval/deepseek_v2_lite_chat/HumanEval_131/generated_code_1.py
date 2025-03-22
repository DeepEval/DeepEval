def method():
    def product_of_odds(n):
        # Convert the number to a string to easily iterate over its digits
        num_str = str(n)
        # Initialize the product to 1
        product = 1
        # Iterate over the digits
        for digit in num_str:
            # Convert the digit to an integer
            digit = int(digit)
            # Check if the digit is odd
            if digit % 2 != 0:
                # Multiply the product by the odd digit
                product *= digit
        # If all digits were even, return 0
        if product == 1:
            return 0
        else:
            return product

    return product_of_odds

# Test the function with a sample input
output = method()(123456)  # Expected output: 21 (1*3*5*7 = 105)
print(output)  # Output should be 105