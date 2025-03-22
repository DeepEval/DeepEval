def method(x, y):
    # Calculate the sum of x and y
    output = x + y
    return output

# Test case
def test_method():
    # Define a test input
    x = 5
    y = 10
    # Expected output is the sum of x and y
    expected_output = 15
    # Call the method function
    result = method(x, y)
    # Check if the result matches the expected output
    assert result == expected_output, f"Test failed: {result} != {expected_output}"
    print("Test passed!")

# Run the test case
test_method()