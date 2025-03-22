def method():
    def is_valid_parentheses(s):
        stack = []
        matching_bracket = {')': '('}
        
        for char in s:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack or stack[-1] != matching_bracket[char]:
                    return False
                stack.pop()
        
        return not stack
    
    # Example usage:
    brackets = "((()))"
    output = is_valid_parentheses(brackets)
    return output

# Test case
print(method())  # Expected output: True