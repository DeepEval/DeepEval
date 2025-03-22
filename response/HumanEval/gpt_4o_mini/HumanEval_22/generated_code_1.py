def method(input_list):
    # Using list comprehension to filter only integers
    output = [item for item in input_list if isinstance(item, int)]
    return output

# Test case for validation
if __name__ == "__main__":
    test_input = [1, 'string', 3.14, 42, True, None, -7, 'another string', 0]
    expected_output = [1, 42, -7, 0]
    
    result = method(test_input)
    
    # Validate the result
    # assert result == expected_output, f"Test failed: {result} != {expected_output}"
    
    # print("Test passed! Output:", result)