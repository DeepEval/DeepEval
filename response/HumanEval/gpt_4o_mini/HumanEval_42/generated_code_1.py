def method(input_list):
    # Increment each element in the input list by 1
    output = [x + 1 for x in input_list]
    return output

# Test case for validation
if __name__ == "__main__":
    test_input = [1, 2, 3, 4, 5]
    expected_output = [2, 3, 4, 5, 6]
    actual_output = method(test_input)
    
    print(f"Test Input: {test_input}")
    print(f"Expected Output: {expected_output}")
    print(f"Actual Output: {actual_output}")
    
    # Validate the output
    # assert actual_output == expected_output, "Test case failed!"
    # print("Test case passed!")