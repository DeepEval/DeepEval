def method(input_list):
    # Convert the list to a set to remove duplicates
    unique_elements = set(input_list)
    # Sort the unique elements
    sorted_unique_elements = sorted(unique_elements)
    return sorted_unique_elements

# Test case
def test_method():
    input_list = [4, 6, 2, 8, 4, 2, 7, 1]
    expected_output = [1, 2, 4, 6, 7, 8]
    assert method(input_list) == expected_output, f"Test failed: expected {expected_output}, got {method(input_list)}"
    print("Test passed!")

# Run the test case
test_method()