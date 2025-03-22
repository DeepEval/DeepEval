import re

def method():
    # Input string
    input_string = "This is a test string."
    
    # Find the last non-space character
    last_non_space_character = re.findlast(r'\S', input_string)
    
    # Check if the character is alphabetical and not part of a word
    if last_non_space_character.isalpha() and not last_non_space_character.group().isspace():
        return True
    else:
        return False

# Test case
test_case = method()
print(test_case)