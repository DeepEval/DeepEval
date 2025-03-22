def method(input_list, substring):
    """
    Filters a list of strings for those containing a given substring.

    Args:
        input_list: A list of strings to filter.
        substring: The substring to search for.

    Returns:
        A new list containing only the strings from input_list that include the substring.
    """
    output = [string for string in input_list if substring in string]
    return output

# Test Case
input_list = ["apple", "banana", "cherry", "date", "elderberry"]
substring = "an"
filtered_list = method(input_list, substring)
print(f"Input List: {input_list}")
print(f"Substring: {substring}")
print(f"Filtered List: {filtered_list}")