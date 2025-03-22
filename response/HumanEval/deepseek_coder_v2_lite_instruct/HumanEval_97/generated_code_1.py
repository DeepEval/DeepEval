def method():
    # Get user input for the two integers
    # num1 = int(input("Enter the first integer: "))
    # num2 = int(input("Enter the second integer: "))
    
    num1 = 123
    num2 = 456

    # Calculate the unit digits
    unit_digit1 = num1 % 10
    unit_digit2 = num2 % 10
    
    # Calculate the product of the unit digits
    product = unit_digit1 * unit_digit2
    
    # Return the final output
    return product

# Test case
output = method()
print("The product of the unit digits is:", output)