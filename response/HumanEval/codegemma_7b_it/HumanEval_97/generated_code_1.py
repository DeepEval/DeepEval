from math import floor

def method(a, b):
    """
    This function takes two integers and returns the product of their unit digits.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of the unit digits of a and b.
    """

    # Extract the unit digits of a and b.
    a_unit_digit = a % 10
    b_unit_digit = b % 10

    # Calculate the product of the unit digits.
    product = a_unit_digit * b_unit_digit

    # Return the product.
    return product


# Test case.
test_case_a = 123
test_case_b = 456
expected_output = 4

output = method(test_case_a, test_case_b)

if output == expected_output:
    print("Test case passed.")
else:
    print("Test case failed.")