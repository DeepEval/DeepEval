from math import floor, ceil

def method(number):
    """
    This function takes a value (string) representing a number and returns the closest integer to it.
    If the number is equidistant from two integers, round it away from zero.
    """
    number = float(number)
    if number >= 0:
        return ceil(number)
    else:
        return floor(number)

# Test case
number = "2.5"
closest_integer = method(number)
print(closest_integer)  # Output: 3