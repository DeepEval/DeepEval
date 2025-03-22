import re

def method(text):
    # Split the string into words using regular expression
    words = re.split(r',|\s+', text)

    # Return the list of words
    return words

# Test case
text = "This is a string, with, punctuation."
words = method(text)
print(words)