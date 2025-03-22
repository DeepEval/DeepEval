def method(s):
    """
    This function takes a string, sorts each word in ascending order of ASCII values, 
    and returns the ordered string. The order of words and blank spaces in the sentence remains the same.

    Args:
        s (str): The input string.

    Returns:
        str: The ordered string.
    """
    # Split the string into words
    words = s.split()

    # Initialize an empty list to store the ordered words
    ordered_words = []

    # Iterate over each word in the list of words
    for word in words:
        # Remove any leading or trailing whitespace from the word
        word = word.strip()
        
        # If the word is not empty
        if word:
            # Sort the characters in the word in ascending order of ASCII values
            ordered_word = ''.join(sorted(word))
            # Append the ordered word to the list of ordered words
            ordered_words.append(ordered_word)
        else:
            # If the word is empty, append an empty string to the list of ordered words
            ordered_words.append('')

    # Join the ordered words back into a string with spaces in between
    ordered_string =''.join(ordered_words)

    # Return the ordered string
    return ordered_string


# Test case
input_string = "Hello World"
print(method(input_string))