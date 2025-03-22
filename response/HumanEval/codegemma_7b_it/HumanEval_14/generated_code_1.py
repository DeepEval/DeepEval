from itertools import accumulate

def method(string):
    """Return list of all prefixes from shortest to longest of the input string.

    Args:
        string (str): The input string.

    Returns:
        list: A list of all prefixes from shortest to longest of the input string.
    """

    # Create a list of all prefixes of the input string.
    prefixes = [''.join(prefix) for prefix in accumulate(string)]

    # Return the prefixes in ascending order of length.
    return sorted(prefixes, key=len)


# Test case.
string = 'hello'
prefixes = method(string)
print(prefixes)