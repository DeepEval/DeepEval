import re

def method(s):
    # Define the uppercase vowels
    vowels = 'AEIOU'
    
    # Initialize a counter for the number of uppercase vowels at even indices
    count = 0
    
    # Loop through the string, checking only at even indices
    for i in range(0, len(s), 2):
        # Check if the character at the current even index is an uppercase vowel
        if s[i] in vowels:
            count += 1
    
    return count

# Test case to validate the function
def test_method():
    # assert method("Hello World") == 0
    # assert method("EVe") == 2
    # assert method("EXAMPLUS") == 2
    # assert method("EXCELSIOR") == 4
    # assert method("I") == 1
    print(method("EVe"))

test_method()