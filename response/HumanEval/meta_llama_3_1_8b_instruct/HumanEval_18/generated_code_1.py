import re

def method(original_string, substring):
    """
    Find how many times a given substring can be found in the original string.
    Count overlapping cases.

    Args:
        original_string (str): The original string to search in.
        substring (str): The substring to search for.

    Returns:
        int: The number of occurrences of the substring in the original string.
    """
    # Use re.finditer to find all occurrences of the substring in the original string
    # The're.finditer' function returns an iterator yielding match objects for every non-overlapping occurrence
    # We use the 'len' function to count the number of occurrences
    occurrences = len(list(re.finditer(re.escape(substring), original_string)))

    return occurrences

# Test case for validation
original_string = "abababab"
substring = "ab"
output = method(original_string, substring)
print("Number of occurrences:", output)