def method(str_list):
    """Concatenates a list of strings into a single string.

    Args:
    str_list: A list of strings.

    Returns:
    A single string containing the concatenated strings.
    """

    output = ""
    for string in str_list:
        output += string

    return output


# Test case
str_list = ["hello", "world", "!"]
output = method(str_list)

if output == "hello world !":
    print("Test passed")
else:
    print("Test failed")