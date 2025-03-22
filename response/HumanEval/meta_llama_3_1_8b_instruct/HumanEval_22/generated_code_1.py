def method(input_list):
    """
    This function filters a given list and returns a new list containing only integers.

    Args:
    input_list (list): The input list that may contain different types of values.

    Returns:
    list: A new list containing only integers from the input list.
    """
    # Check if input is a list
    if not isinstance(input_list, list):
        raise ValueError("Input must be a list")

    # Use list comprehension to filter integers
    output = [value for value in input_list if isinstance(value, int)]
    
    return output

# Test case
input_list = [1, 2, 'a', 3, 4.5, 'b', 5, 6.7, 7, 'c', 8, 9]
print(method(input_list))  # Output: [1, 2, 3, 5, 7, 8, 9]

# Test case with non-list input
try:
    print(method('non-list input'))
except ValueError as e:
    print(e)  # Output: Input must be a list