from typing import Union

def method(num: int) -> str:
    """
    Converts an integer to its roman numeral equivalent.

    Args:
        num: A positive integer between 1 and 1000.

    Returns:
        The roman numeral equivalent of the input integer as a lowercase string.
    """

    if not (1 <= num <= 1000):
        raise ValueError("Input integer must be between 1 and 1000.")

    roman_numerals = {
        1000: "m",
        900: "cm",
        500: "d",
        400: "cd",
        100: "c",
        90: "xc",
        50: "l",
        40: "xl",
        10: "x",
        9: "ix",
        5: "v",
        4: "iv",
        1: "i",
    }

    output = ""

    for value, numeral in roman_numerals.items():
        while num >= value:
            output += numeral
            num -= value

    return output.lower()

# Test case
test_case = 123
expected_output = "CXXIII"
actual_output = method(test_case)

if actual_output == expected_output:
    print("Test case passed.")
else:
    print("Test case failed. Expected:", expected_output, "Actual:", actual_output)