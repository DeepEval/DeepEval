def method(input_str):
    # Split the input string into groups based on spaces
    groups = input_str.split()
    
    # Initialize a list to store the maximum depths for each group
    max_depths = []
    
    # Iterate through each group
    for group in groups:
        current_depth = 0  # Current depth of nesting
        max_depth = 0      # Maximum depth found for this group
        
        # Iterate through each character in the group
        for char in group:
            if char == '(':  # Opening parenthesis
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':  # Closing parenthesis
                current_depth -= 1
            
            # If current depth becomes negative, parentheses are unbalanced
            if current_depth < 0:
                break
        
        # Append the maximum depth found for the group
        max_depths.append(max_depth)
    
    return max_depths

# Test case
test_input = "(()()) ((())) (()(())) (()"
output = method(test_input)
print(output)  # Expected output: [2, 3, 2, 1]