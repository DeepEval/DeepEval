import re

def method(input_string):
    # Use regex to split by commas or spaces, and filter out any empty strings
    words = re.split(r'[,\s]+', input_string.strip())
    return [word for word in words if word]  # Filter out any empty strings

# Test case for validation
test_input = "apple, banana orange,grape  pear"
output = method(test_input)
print(output)  # Expected output: ['apple', 'banana', 'orange', 'grape', 'pear']