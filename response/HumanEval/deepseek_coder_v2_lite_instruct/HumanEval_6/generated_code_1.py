def method():
    # Test case
    test_cases = [
        "(()())",
        "((()))",
        "((())())",
        "()",
        "()()",
        "(())",
        "((())())"
    ]
    
    output = []
    
    for test in test_cases:
        max_depth = 0
        current_depth = 0
        for char in test:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        output.append(max_depth)
    
    return output

# Run the method and print the output
print(method())