import re

def method(input_string):
    """
    Split a string into words separated by commas or spaces.

    Args:
    input_string (str): The input string to be split.

    Returns:
    list: A list of words.
    """
    # Use regular expression to split the string into words
    words = re.findall(r'\w+', input_string)
    
    return words

# Test case
input_string = "Hello, World! Python is fun, isn't it?"
output = method(input_string)
print(output)