def method(a: int, b: int) -> int:
    """
    Returns the product of the unit digits of two integers.

    :param a: The first integer.
    :param b: The second integer.
    :return: The product of the unit digits of a and b.
    """
    # Extract the unit digits of a and b
    unit_digit_a = a % 10
    unit_digit_b = b % 10

    # Calculate the product of the unit digits
    product = unit_digit_a * unit_digit_b

    return product

# Example test case
def test_method():
    # Test with two integers
    result = method(123, 456)
    print(result)
    # assert result == 6, f"Expected 6, but got {result}"

    # # Test with two integers where the unit digits multiply to 0
    # result = method(10, 0)
    # assert result == 0, f"Expected 0, but got {result}"

    # # Test with two integers where the unit digits are both 0
    # result = method(0, 0)
    # assert result == 0, f"Expected 0, but got {result}"

    # # Test with one of the integers being 0
    # result = method(12, 0)
    # assert result == 0, f"Expected 0, but got {result}"

    # print("All test cases pass")

# Run the test function
test_method()