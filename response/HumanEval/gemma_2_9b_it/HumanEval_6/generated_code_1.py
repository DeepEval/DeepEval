def method(input_string):
    """
    Determines the deepest level of nesting for parentheses in a string.

    Args:
        input_string (str): A string containing groups of nested parentheses separated by spaces.

    Returns:
        list: A list containing the deepest nesting level for each group of parentheses.
    """

    output = []
    for group in input_string.split():
        max_depth = 0
        current_depth = 0
        for char in group:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        output.append(max_depth)
    return output

# Test case
input_string = "(()()) ((()))"
result = method(input_string)
print(f"Input: {input_string}")
print(f"Output: {result}")