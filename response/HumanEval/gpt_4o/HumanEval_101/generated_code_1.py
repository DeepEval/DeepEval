import re

def method(input_string):
    # Use regular expression to split by commas or spaces
    words = re.split(r'[,\s]+', input_string)
    
    # Filter out any empty strings that might occur due to leading/trailing spaces or commas
    words = [word for word in words if word]
    
    return words

# Test case
test_input = "hello, world  ,this is,  a test"
expected_output = ['hello', 'world', 'this', 'is', 'a', 'test']
output = method(test_input)

# Assertion to validate the output
assert output == expected_output, f"Test failed: {output} != {expected_output}"

print("Test passed. Output:", output)