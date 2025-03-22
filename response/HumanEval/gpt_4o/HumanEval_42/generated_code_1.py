def method(input_list):
    # Increment each element in the input_list by 1
    output = [element + 1 for element in input_list]
    return output

# Test case
def test_method():
    # Define a test input list
    test_input = [1, 2, 3, 4, 5]
    # Expected output is each element incremented by 1
    expected_output = [2, 3, 4, 5, 6]
    
    # Call the method with the test input
    result = method(test_input)
    
    # Assert to check if the result matches the expected output
    assert result == expected_output, f"Test failed: expected {expected_output}, got {result}"
    print("Test passed!")

# Run the test case
test_method()