def method():
    numbers = [1, 2, 3, 4, 5]
    delimiter = 0
    output = []

    for i in range(len(numbers)):
        output.append(numbers[i])
        if i < len(numbers) - 1:
            output.append(delimiter)

    return output

# Test case
def test_method():
    result = method()
    expected = [1, 0, 2, 0, 3, 0, 4, 0, 5]
    # assert result == expected, f"Expected {expected}, but got {result}"

    # print("Test passed!")

# Run the test
test_method()