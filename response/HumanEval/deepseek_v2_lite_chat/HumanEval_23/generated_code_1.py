def method():
    # Example input
    input_string = "Hello, World!"
    
    # Using len() to get the length of the string
    length = len(input_string)
    
    # Return the length
    return length

# Test case to validate the function
def test_method():
    # Expected output for the example input
    expected_output = 13
    
    # Call the method() function and store the result
    result = method()
    print(result)
    
    # Compare the result with the expected output
    # assert expected_output == result, f"Expected output: {expected_output}, but got: {result}"
    
    print("Test passed.")

# Run the test case
test_method()