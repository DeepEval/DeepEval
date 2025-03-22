def method():
    brackets = "()(())"
    stack = []
    for char in brackets:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:  # If stack is empty, no matching opening bracket
                return False
            stack.pop()
    return not stack  # True if stack is empty (all brackets matched)

output = method()
print(output)  # Output: True