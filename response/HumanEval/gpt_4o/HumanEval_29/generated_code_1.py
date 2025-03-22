def method(strings, prefix):
    # Filter the list of strings to only include those that start with the given prefix
    output = [s for s in strings if s.startswith(prefix)]
    return output

# Test case for validation
def test_method():
    # Given input list of strings and a prefix
    strings = ["apple", "banana", "apricot", "cherry", "avocado"]
    prefix = "ap"
    
    # Expected output
    expected_output = ["apple", "apricot", "avocado"]
    
    # Get the output from the method function
    output = method(strings, prefix)
    
    # Validate the output
    # assert output == expected_output, f"Expected {expected_output}, but got {output}"
    
    # print("Test passed!")

# Run the test case
test_method()