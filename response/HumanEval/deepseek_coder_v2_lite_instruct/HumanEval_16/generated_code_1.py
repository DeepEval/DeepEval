def method():
    # Example string
    input_string = "Hello, World!"
    
    # Convert the string to lowercase to ignore case
    lowercase_string = input_string.lower()
    
    # Use a set to store distinct characters
    distinct_characters = set(lowercase_string)
    
    # The number of distinct characters
    output = len(distinct_characters)
    
    return output

# Test case
# def test_method():
#     assert method() == 10, "Test case failed: Expected 10 distinct characters."
#     print("Test case passed.")

# # Run the test case
# test_method()

# Run the method
method()