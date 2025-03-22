import re

def method(s):
    """
    This function calculates the deepest level of nesting of parentheses in each group of a string.
    
    Parameters:
    s (str): A string with multiple groups of nested parentheses separated by spaces.
    
    Returns:
    list: A list of integers representing the deepest level of nesting of parentheses in each group.
    """

    # Split the input string into groups
    groups = s.split()

    # Initialize an empty list to store the results
    output = []

    # Iterate over each group
    for group in groups:
        # Initialize a counter for the level of nesting
        level = 0
        
        # Initialize a flag to indicate if we are inside a group
        inside_group = False
        
        # Initialize a counter to store the maximum level of nesting
        max_level = 0
        
        # Iterate over each character in the group
        for char in group:
            # If the character is an opening parenthesis, increment the level
            if char == '(':
                level += 1
                # If we are not inside a group, set the flag to True
                if not inside_group:
                    inside_group = True
            # If the character is a closing parenthesis, decrement the level
            elif char == ')':
                level -= 1
                # If the level is 0, we are no longer inside a group
                if level == 0:
                    inside_group = False
            # Update the maximum level of nesting
            max_level = max(max_level, level)
        
        # Append the maximum level of nesting to the output list
        output.append(max_level)

    return output

# Test case
s = "((())) (()()) (())()"
output = method(s)
print(output)  # Expected output: [3, 2, 1]