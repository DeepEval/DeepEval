import re

def is_palindrome(text):
    text = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
    return text == text[::-1]

def method():
    input_text = "A man, a plan, a canal: Panama"
    output = is_palindrome(input_text)

    return output

# Test case
test_input = "racecar"
test_output = is_palindrome(test_input)
print(test_output)  # Output: True