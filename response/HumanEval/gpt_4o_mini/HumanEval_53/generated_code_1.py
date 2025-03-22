def method(x, y):
    # Add the two numbers x and y
    output = x + y
    return output

# Test case for validation
if __name__ == "__main__":
    # Example test case
    test_x = 5
    test_y = 7
    expected_output = 12
    result = method(test_x, test_y)
    
    # Validate the result
    # assert result == expected_output, f"Test failed: {result} != {expected_output}"
    # print(f"Test passed: {result} == {expected_output}")