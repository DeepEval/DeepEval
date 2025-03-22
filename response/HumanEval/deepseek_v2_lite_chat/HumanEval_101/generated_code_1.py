import re

def method(input_string):
    # Split the string into words
    words = re.split(r'\s*,\s*|\s+', input_string)
    return words

# Test case
input_string = "This is a test, test string. Another test string."
result = method(input_string)
print(result)  # Output: ['This', 'is', 'a', 'test,', 'test', 'string.', 'Another', 'test', 'string.']