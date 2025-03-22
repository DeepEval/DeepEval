def validate_brackets(brackets):
    """
    Validate whether every opening bracket in a string has a corresponding closing bracket.

    Args:
        brackets (str): A string of "(" and ")".

    Returns:
        bool: True if every opening bracket has a corresponding closing bracket, False otherwise.
    """
    stack = []
    for bracket in brackets:
        if bracket == "(":
            # Push opening bracket onto the stack
            stack.append(bracket)
        elif bracket == ")":
            # If stack is empty, return False
            if not stack:
                return False
            # Pop opening bracket from the stack
            stack.pop()
    # If stack is not empty at the end, return False
    return not stack


# Test case
brackets = "((()))"
output = validate_brackets(brackets)
print("Input:", brackets)
print("Output:", output)

brackets = "(()"
output = validate_brackets(brackets)
print("Input:", brackets)
print("Output:", output)

brackets = "())"
output = validate_brackets(brackets)
print("Input:", brackets)
print("Output:", output)

brackets = ""
output = validate_brackets(brackets)
print("Input:", brackets)
print("Output:", output)

brackets = "(())"
output = validate_brackets(brackets)
print("Input:", brackets)
print("Output:", output)