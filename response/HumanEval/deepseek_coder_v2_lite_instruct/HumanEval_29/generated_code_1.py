def method():
    # Import necessary packages if needed
    # For this task, no additional packages are needed
    
    # Define the input list of strings and the prefix to filter by
    input_list = ["apple", "banana", "apricot", "avocado", "blueberry", "blackberry"]
    prefix = "ap"
    
    # Filter the list based on the prefix
    output = [s for s in input_list if s.startswith(prefix)]
    
    return output

# # Test case for validation
# def test_method():
#     result = method()
#     expected = ["apple", "apricot", "avocado"]
#     assert result == expected, f"Expected {expected}, but got {result}"
#     print("Test passed!")

# # Run the test case
# test_method()

# Run the method
method()