def method():
    def helper(s, depth):
        max_depth = 0
        current_depth = 0
        for char in s:
            if char == '(':
                current_depth += 1
                if current_depth > max_depth:
                    max_depth = current_depth
            elif char == ')':
                current_depth -= 1
        return max_depth

    # Input string
    input_str = "((()))"
    
    # Find the maximum depth of nesting
    max_depth = helper(input_str, 0)
    
    return max_depth

# Test case
output = method()
print("The deepest level of nesting is:", output)