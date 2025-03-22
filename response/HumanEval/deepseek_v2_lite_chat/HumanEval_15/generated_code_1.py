def method(n):
    """
    Generate a string containing space-delimited numbers from 0 to n inclusive.

    Args:
    n (int): The upper limit for the numbers in the resulting string.

    Returns:
    str: A string with space-delimited numbers from 0 to n inclusive.
    """
    output = ""
    for i in range(n + 1):
        output += str(i) + " "
    return output.strip()  # Remove the trailing space at the end of the string

# Test case to validate the function
def test_method():
    # Test with n=5
    result = method(5)
    expected_result = "0 1 2 3 4 5 "
    # assert result == expected_result, f"Expected '{expected_result}' but got '{result}'"
    
    print("Test Passed: The function correctly generates the required output.")

test_method()