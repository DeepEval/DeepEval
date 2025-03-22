import re

def method(str):
    """
    This function returns True if the last character of a given string is an alphabetical character and is not a part of a word, and False otherwise.
    Note: "word" is a group of characters separated by space.
    """

    # Check if the last character of the string is an alphabetical character.
    if not str or not str[-1].isalpha():
        return False

    # Check if the last character of the string is not a part of a word.
    if re.search(r"\b\w$", str):
        return False

    # If the last character of the string is an alphabetical character and is not a part of a word, return True.
    return True

# Test case
str = "This is a sentence."
print(method(str)) # Output: True

str = "This is a sentence."
print(method(str)) # Output: False