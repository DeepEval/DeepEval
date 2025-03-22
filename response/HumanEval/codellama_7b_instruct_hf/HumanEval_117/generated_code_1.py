# Import necessary packages
import re

def method(s, n):
    """
    Given a string s and a natural number n, returns a list of all words from string s that contain exactly n consonants, in order these words appear in the string s.
    If the string s is empty then the function should return an empty list.
    Note: you may assume the input string contains only letters and spaces.

    Args:
        s: The input string.
        n: The number of consonants in the words to return.

    Returns:
        A list of words from string s that contain exactly n consonants, in order these words appear in the string s.
        If the string s is empty then the function should return an empty list.
    """

    # Check if the input string is empty
    if not s:
        return []

    # Initialize the output list
    output = []

    # Split the string into words
    words = s.split()

    # Iterate over the words
    for word in words:
        # Count the number of consonants in the word
        num_consonants = len(re.findall(r'[aeiou]', word, flags=re.IGNORECASE))

        # If the number of consonants is equal to n, add the word to the output list
        if num_consonants == n:
            output.append(word)

    # Return the output list
    return output

# Test case
s = "This is a string with 3 consonants in it"
n = 3

print(method(s, n))