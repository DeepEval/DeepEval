def method(lst):
    """
    This function takes a list of numbers as input and returns a new list containing only the positive numbers.

    Args:
        lst (list): A list of numbers.

    Returns:
        list: A list of positive numbers.
    """
    # Filter the list to include only positive numbers
    positive_numbers = [num for num in lst if num > 0]
    
    return positive_numbers


# Test case
numbers = [-1, 2, 0, -3, 4, 5, -6]
output = method(numbers)
print("Output:", output)