def method(input_string):
    # Flip the case of each character in the input string
    output = input_string.swapcase()
    return output

# Test case for validation
def test_method():
    # Test input
    input_string = "Hello, World!"
    # Expected output: "hELLO, wORLD!"
    expected_output = "hELLO, wORLD!"
    
    # Run the method
    result = method(input_string)
    
    # Check if the result matches the expected output
    assert result == expected_output, f"Test failed: expected {expected_output}, got {result}"
    print("Test passed!")

# Run the test case
test_method()