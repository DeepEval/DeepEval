import re

def method(input_string):
    # Remove spaces from the input string
    input_string = input_string.replace(" ", "")

    # Regex pattern to match groups of nested parentheses
    pattern = r"\([^()]*\)"

    # Find all groups of parentheses in the input string
    groups = re.findall(pattern, input_string)

    # Return the list of groups
    return groups

# Test case
input_string = "(hello world)([this is a nested group])"
groups = method(input_string)

# Print the groups
print(groups)