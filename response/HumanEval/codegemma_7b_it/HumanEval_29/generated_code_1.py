import re

def method(input_list, prefix):
    # Use regular expressions to filter the list of strings
    output = [string for string in input_list if re.match(prefix, string)]
    return output

# Test case
input_list = ["apple", "banana", "orange", "pineapple", "grapefruit"]
prefix = "apple"
output = method(input_list, prefix)
print(output)