def next_smallest(numbers):
    if len(numbers) < 2:
        return None

    # Convert list to a set to get unique numbers
    unique_numbers = set(numbers)

    # If there are fewer than 2 unique numbers, return None
    if len(unique_numbers) < 2:
        return None

    # Sort the unique numbers and get the second smallest
    sorted_numbers = sorted(unique_numbers)
    return sorted_numbers[1]

def method():
    # Test cases
    test_cases = [
        ([1, 2, 3, 4, 5], 2),
        ([5, 1, 4, 3, 2], 2),
        ([], None),
        ([1, 1], None)
    ]

    # Validate the test cases
    for i, (input_list, expected_output) in enumerate(test_cases):
        assert next_smallest(input_list) == expected_output, f"Test case {i + 1} failed"

    # If all tests pass, return success message
    return "All test cases passed."

# Execute method
print(method())