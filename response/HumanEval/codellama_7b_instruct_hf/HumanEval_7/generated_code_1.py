import re

def method(input_list, substring):
    """
    This method filters an input list of strings for ones that contain a given substring.

    Args:
        input_list: The input list of strings.
        substring: The substring to look for.

    Returns:
        A new list of strings that contain the substring.
    """

    # Use a regular expression to filter the strings.
    filtered_list = [string for string in input_list if re.search(substring, string)]

    return filtered_list

# Test case
input_list = ["apple", "banana", "orange", "grapefruit", "pineapple"]
substring = "an"

filtered_list = method(input_list, substring)

# Print the filtered list
print(filtered_list)