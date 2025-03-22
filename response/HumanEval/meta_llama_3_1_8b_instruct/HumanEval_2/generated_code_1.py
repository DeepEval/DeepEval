import math

def method(number):
    """
    Decompose a positive floating point number into its decimal part.

    Args:
        number (float): A positive floating point number.

    Returns:
        float: The decimal part of the given number.
    """
    # Check if the number is positive
    if number <= 0:
        raise ValueError("The input number must be a positive floating point number.")

    # Get the integer part using math.floor()
    integer_part = math.floor(number)

    # Calculate the decimal part by subtracting the integer part from the original number
    decimal_part = number - integer_part

    return decimal_part

# Test case
number = 3.75
output = method(number)
print(f"The decimal part of {number} is {output}")