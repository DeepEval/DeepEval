# importing necessary packages
import re

# define the method function
def method(msg):
    # converting the message to lowercase
    msg = msg.lower()

    # creating a dictionary of vowels and their replacements
    vowels = {'a': 'c', 'e': 'g', 'i': 'k', 'o': 'q', 'u': 's'}

    # replacing vowels with their replacements
    for vowel in vowels:
        msg = msg.replace(vowel, vowels[vowel])

    # swapping the case of all letters
    output = ''.join([char.lower() if char.isupper() else char.upper() for char in msg])

    return output

# test case
msg = "Hello World!"
output = method(msg)
print(output)