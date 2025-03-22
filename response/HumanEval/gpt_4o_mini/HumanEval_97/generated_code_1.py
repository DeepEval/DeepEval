def method(num1, num2):
    # Extract the unit digits using modulus operator
    unit_digit1 = num1 % 10
    unit_digit2 = num2 % 10
    
    # Calculate the product of the unit digits
    product = unit_digit1 * unit_digit2
    
    return product

# Test case for validation
if __name__ == "__main__":
    # Example test case: 23 and 45
    # Unit digits are 3 and 5, so the product should be 15
    test_case_num1 = 23
    test_case_num2 = 45
    output = method(test_case_num1, test_case_num2)
    print(f'The product of the unit digits of {test_case_num1} and {test_case_num2} is: {output}')