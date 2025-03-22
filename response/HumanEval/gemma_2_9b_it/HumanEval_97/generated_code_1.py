def method(num1, num2):
    """
    This function takes two integers and returns the product of their unit digits.

    Args:
        num1 (int): The first integer.
        num2 (int): The second integer.

    Returns:
        int: The product of the unit digits of the two integers.
    """
    unit_digit1 = num1 % 10
    unit_digit2 = num2 % 10
    return unit_digit1 * unit_digit2

# Test case
num1 = 12
num2 = 34
output = method(num1, num2)
print(f"The product of the unit digits of {num1} and {num2} is: {output}")