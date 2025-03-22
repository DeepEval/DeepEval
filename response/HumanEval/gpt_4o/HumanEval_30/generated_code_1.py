def method(numbers):
    # Filter the list to include only positive numbers
    output = [num for num in numbers if num > 0]
    return output

# Test case for validation
if __name__ == "__main__":
    test_list = [-10, 0, 5, -3, 2, 9, -1]
    expected_output = [5, 2, 9]
    
    result = method(test_list)
    print("Output:", result)
    assert result == expected_output, f"Test failed: Expected {expected_output}, but got {result}"
    print("Test passed.")