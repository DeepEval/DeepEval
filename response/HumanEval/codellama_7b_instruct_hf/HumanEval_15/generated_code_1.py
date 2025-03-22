from typing import List

def method(n: int) -> str:
    """
    Returns a string containing space-delimited numbers starting from 0 upto n inclusive.

    Args:
        n: The upper limit of the range of numbers to include in the string.

    Returns:
        A string containing space-delimited numbers from 0 to n inclusive.
    """

    numbers = []
    for i in range(n + 1):
        numbers.append(str(i))

    output = " ".join(numbers)
    return output

# Test case
n = 5
expected_output = "0 1 2 3 4 5"
actual_output = method(n)

# assert actual_output == expected_output, f"Expected {expected_output}, but got {actual_output}"

# print("Test case passed.")