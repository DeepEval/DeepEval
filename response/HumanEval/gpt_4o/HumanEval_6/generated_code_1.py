def method(parentheses_string):
    # Split the input string into individual groups based on spaces
    groups = parentheses_string.split()
    
    # Function to calculate the depth of nesting for a single group
    def calculate_max_depth(group):
        max_depth = 0
        current_depth = 0
        
        for char in group:
            if char == '(':
                current_depth += 1
                # Update max_depth if current_depth exceeds it
                if current_depth > max_depth:
                    max_depth = current_depth
            elif char == ')':
                current_depth -= 1
        
        return max_depth

    # Calculate the maximum depth for each group
    output = [calculate_max_depth(group) for group in groups]
    return output

# Test case for validation
test_input = "(()()) ((())) () (())"
expected_output = [2, 3, 1, 2]
assert method(test_input) == expected_output, "Test case failed!"

# If no assertion error is raised, the test case is successful
print("Test case passed!")