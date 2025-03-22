def method(input_list):
    """
    This function returns the maximum element in the given list.

    Parameters:
    input_list (list): A list of elements.

    Returns:
    max_element: The maximum element in the list.
    """
    return max(input_list)

# Test case for validation
input_list = [10, 20, 30, 40, 50]
output = method(input_list)
print("Input List:", input_list)
print("Maximum Element:", output)