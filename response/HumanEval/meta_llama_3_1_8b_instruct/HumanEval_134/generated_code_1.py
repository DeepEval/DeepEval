def method(s):
    """
    Checks if the last character of a given string is an alphabetical character and is not part of a word.

    Args:
    s (str): The input string.

    Returns:
    bool: True if the last character is an alphabetical character and not part of a word, False otherwise.
    """
    # Check if input is a string
    if not isinstance(s, str):
        raise TypeError("Input must be a string.")

    # Split the string into words
    words = s.split()

    # Check if the string is not empty
    if len(words) == 0:
        # If the string is empty, return False
        return False

    # Get the last word in the string
    last_word = words[-1]

    # Check if the last character of the last word is an alphabetical character
    if last_word and last_word[-1].isalpha():
        # Check if the last character is the only character in the last word
        if len(last_word) == 1:
            # If it is, return True
            return True

    # If any of the above conditions are not met, return False
    return False

# Test case
input_string = "Hello World"
output = method(input_string)
print("Is the last character of '{}' an alphabetical character and not part of a word? : {}".format(input_string, output))