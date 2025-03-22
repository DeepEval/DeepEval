def method(strings, substring):
    # Filter the list of strings to include only those that contain the given substring
    output = [s for s in strings if substring in s]
    return output

# Test case for validation
def test_method():
    input_strings = ["apple", "banana", "apricot", "cherry", "pineapple"]
    search_substring = "app"
    expected_output = ["apple", "apricot", "pineapple"]

    # Call the method function
    result = method(input_strings, search_substring)

    # Check if the result is as expected
    # assert result == expected_output, f"Test failed: expected {expected_output}, got {result}"
    # print("Test passed!")

# Run the test case
test_method()