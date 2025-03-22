def is_nested(s):
    # To check for nesting, we need at least two '[' and one ']' that can close a bracket.
    open_count = 0
    close_count = 0
    nested_found = False
    
    for char in s:
        if char == '[':
            open_count += 1
            # If we already have an open bracket and find another, we can potentially have nesting
            if open_count > 1:
                nested_found = True
        elif char == ']':
            close_count += 1
            # If we have more closing brackets than opening brackets, it's invalid
            if close_count > open_count:
                return False

    # A valid sequence must have at least one nested bracket and must be balanced
    return nested_found and open_count == close_count

def method():
    # Example test cases
    test_cases = [
        ('[[]]', True),
        ('[]]]]]]][[[[[]', False),
        ('[][]', False),
        ('[]', False),
        ('[[][]]', True),
        ('[[]][[', True),
    ]
    
    output = {}
    
    for input_str, expected in test_cases:
        result = is_nested(input_str)
        output[input_str] = (result == expected)

    return output

# Call the method to get the output
output = method()
print(output)