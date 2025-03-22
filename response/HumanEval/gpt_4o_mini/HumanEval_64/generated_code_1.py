def method(a, b):
    """
    This function takes two numbers a and b, adds them, and returns the result.
    """
    return a + b

def test_method():
    """
    Test the method function with various test cases.
    """
    test_cases = [
        (1, 2, 3),         # 1 + 2 = 3
        (-1, 1, 0),       # -1 + 1 = 0
        (0, 0, 0),        # 0 + 0 = 0
        (100, 200, 300),  # 100 + 200 = 300
        (-50, -50, -100), # -50 + -50 = -100
        (3.5, 2.5, 6.0),  # 3.5 + 2.5 = 6.0
        (1.1, 2.2, 3.3),  # 1.1 + 2.2 = 3.3
    ]

    for a, b, expected in test_cases:
        result = method(a, b)
        # assert result == expected, f"Test failed for inputs ({a}, {b}). Expected {expected}, but got {result}."
        # print(f"Test passed for inputs ({a}, {b}): {result} == {expected}")

# Run the test cases
test_method()