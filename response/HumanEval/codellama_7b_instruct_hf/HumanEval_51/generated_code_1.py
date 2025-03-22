import re

def remove_vowels(text):
    # Remove all vowels using regular expressions
    output = re.sub(r'[aeiouAEIOU]', '', text)
    
    return output

# Test case
text = "This is a test case."
result = remove_vowels(text)
print(result)  # Output: Ths s s tst csc.