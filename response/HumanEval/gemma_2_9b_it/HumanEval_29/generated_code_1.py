def method(input_list, prefix):
    """Filters a list of strings for those that start with a given prefix.

    Args:
        input_list: A list of strings.
        prefix: The prefix to filter by.

    Returns:
        A list of strings from the input list that start with the given prefix.
    """
    output = [string for string in input_list if string.startswith(prefix)]
    return output

# Test case
input_list = ["apple", "banana", "apricot", "avocado", "orange"]
prefix = "ap"
filtered_list = method(input_list, prefix)
print(f"Input list: {input_list}")
print(f"Filtered list (starting with '{prefix}'): {filtered_list}")