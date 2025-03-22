def method(values):
    # Filter the list to include only integers
    output = [value for value in values if isinstance(value, int)]
    return output

# Test case for validation
if __name__ == "__main__":
    test_values = [1, 'a', 3.5, 2, 'hello', 4, 5.6, -1, 'world', 0]
    expected_output = [1, 2, 4, -1, 0]
    result = method(test_values)
    print("Output:", result)
    assert result == expected_output, "Test case failed!"
    print("Test case passed!")