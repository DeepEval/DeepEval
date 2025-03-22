def method(numbers, delimiter):
    output = []
    for i in range(len(numbers)):
        output.append(numbers[i])
        if i < len(numbers) - 1:
            output.append(delimiter)
    return output

# Test case for validation
def test_method():
    numbers = [1, 2, 3, 4]
    delimiter = 0
    expected_output = [1, 0, 2, 0, 3, 0, 4]
    result = method(numbers, delimiter)
    assert result == expected_output, f"Test failed: expected {expected_output}, got {result}"
    print("Test passed!")

# Run the test
test_method()