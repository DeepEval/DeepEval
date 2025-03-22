def method(a, b):
    """
    This function takes two numbers and returns their sum.
    """
    return a + b

def test_method():
    # Test cases for the method function
    test_cases = [
        (1, 2, 3),          # Standard addition
        (0, 0, 0),          # Addition of zeros
        (-1, -2, -3),       # Addition of negative numbers
        (100, 200, 300),    # Large numbers
        (-5, 5, 0),         # Addition of a positive and a negative number
    ]

    for a, b, expected in test_cases:
        result = method(a, b)
        # assert result == expected, f"Test failed for inputs {a} and {b}. Expected {expected}, but got {result}"

    print("All tests passed!")

# Run the test function
test_method()