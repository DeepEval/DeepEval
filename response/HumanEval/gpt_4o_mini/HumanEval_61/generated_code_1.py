def method(brackets):
    # Initialize a stack to keep track of opening brackets
    stack = []
    
    # Iterate through each character in the string
    for char in brackets:
        # If we encounter an opening bracket, push it onto the stack
        if char == '(':
            stack.append(char)
        # If we encounter a closing bracket
        elif char == ')':
            # If stack is not empty, pop the last opening bracket
            # If stack is empty, it means there is no matching opening bracket
            if stack:
                stack.pop()
            else:
                return False
    
    # If the stack is empty, all opening brackets had matching closing brackets
    return len(stack) == 0

# Example test case
test_case = "(()())"
output = method(test_case)
print(output)  # Should print True

# Additional test cases for validation
print(method("((())"))    # Should print False
print(method("()()()"))    # Should print True
print(method(")("))        # Should print False
print(method("(()())()"))  # Should print True