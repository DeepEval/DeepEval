def method(prefix, input_list):
    """
    Filter a list of strings for ones that start with a given prefix.

    Args:
        prefix (str): The prefix to filter by.
        input_list (list): The list of strings to filter.

    Returns:
        list: A list of strings that start with the given prefix.
    """
    return [string for string in input_list if string.startswith(prefix)]

# Test case
input_list = ["apple", "banana", "cherry", "apricot", "avocado"]
prefix = "a"
output = method(prefix, input_list)
print(output)  # Output: ["apple", "apricot", "avocado"]