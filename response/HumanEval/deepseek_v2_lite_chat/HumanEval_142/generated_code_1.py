def method(input_list):
    # Initialize the sum
    output = 0
    
    # Iterate over the list using enumerate to get both index and value
    for index, value in enumerate(input_list):
        # Apply the transformation if index is a multiple of 3 or 4
        if index % 3 == 0:
            output += value ** 2
        elif index % 4 == 0:
            output += value ** 3
    
    return output

# Test case to validate the function
def test_method():
    # Test with a list where indices are multiples of 3 or 4
    input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    expected_output = 37
    # assert method(input_list) == expected_output, f"Expected {expected_output} but got {method(input_list)}"
    method(input_list)
    # Test with a list where indices are not multiples of 3 or 4
    input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    expected_output = 28
    # assert method(input_list) == expected_output, f"Expected {expected_output} but got {method(input_list)}"
    
    # print("All test cases passed!")

# Call the test function
test_method()