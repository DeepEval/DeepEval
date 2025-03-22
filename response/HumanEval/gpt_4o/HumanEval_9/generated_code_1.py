def method(input_list):
    if not input_list:
        return []

    rolling_max = []
    current_max = float('-inf')

    for number in input_list:
        current_max = max(current_max, number)
        rolling_max.append(current_max)

    return rolling_max

# Test case for validation
test_input = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
expected_output = [3, 3, 4, 4, 5, 9, 9, 9, 9, 9, 9]

assert method(test_input) == expected_output, "Test case failed!"

print("Test case passed!")