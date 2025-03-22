def increment_list(input_list):
    """
    Return a new list with elements incremented by 1.

    Args:
        input_list (list): The input list to be incremented.

    Returns:
        list: A new list with elements incremented by 1.
    """
    return [element + 1 for element in input_list]


# Example usage:
input_list = [1, 2, 3, 4, 5]
output = increment_list(input_list)
print(output)  # Output: [2, 3, 4, 5, 6]

# Test case for validation:
def test_increment_list():
    input_list = [1, 2, 3, 4, 5]
    expected_output = [2, 3, 4, 5, 6]
    actual_output = increment_list(input_list)
    # assert actual_output == expected_output, f"Expected {expected_output} but got {actual_output}"

test_increment_list()
print("Test passed!")