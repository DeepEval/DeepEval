def method(input_list):
    """
    Returns the longest string from a list of strings.
    In case of multiple strings with the same length, returns the first one.
    Returns None if the input list is empty.

    Args:
        input_list (list): A list of strings.

    Returns:
        str: The longest string from the input list, or None if the list is empty.
    """
    if not input_list:  # Check if the list is empty
        return None

    return max(input_list, key=len)  # Return the longest string

# Test case
input_list = ["apple", "banana", "cherry", "date", "elderberry"]
print(method(input_list))  # Output: "elderberry"

input_list = []
print(method(input_list))  # Output: None

input_list = ["apple", "banana", "cherry"]
print(method(input_list))  # Output: "banana"