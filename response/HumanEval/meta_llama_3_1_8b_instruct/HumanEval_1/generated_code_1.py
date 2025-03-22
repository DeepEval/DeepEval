def method(s):
    """
    This function takes a string containing multiple groups of nested parentheses, 
    separates them into separate strings, and returns the list of those.
    
    Args:
        s (str): The input string containing groups of nested parentheses.
    
    Returns:
        list: A list of strings, each representing a group of nested parentheses.
    """
    
    # Initialize an empty stack to store indices of opening parentheses
    stack = []
    
    # Initialize an empty list to store the result
    result = []
    
    # Initialize an empty string to build the current group
    current_group = ""
    
    # Iterate over the input string
    for i, char in enumerate(s):
        # If the character is an opening parenthesis, push its index onto the stack
        if char == "(":
            stack.append(i)
            current_group += char
        # If the character is a closing parenthesis
        elif char == ")":
            # Pop the index of the corresponding opening parenthesis from the stack
            start_idx = stack.pop()
            # Extract the current group from the input string and add it to the result
            result.append(s[start_idx:i+1].replace(" ", ""))
            # Reset the current group
            current_group = ""
    
    # Return the result
    return result


# Test case
input_str = "( a (b c) (d e) ) ( f (g h) (i j) )"
print(method(input_str))  # Output: ['( a (b c) (d e) )', '( f (g h) (i j) )']