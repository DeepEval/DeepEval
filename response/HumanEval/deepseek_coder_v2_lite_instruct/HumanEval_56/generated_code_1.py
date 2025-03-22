def method():
    brackets = "><<><>>"
    stack = []
    
    for bracket in brackets:
        if bracket == '<':
            stack.append(bracket)
        elif bracket == '>':
            if not stack:
                return False
            stack.pop()
    
    return len(stack) == 0

# Test case
test_brackets = "><<><>>"
output = method()
print(f"Test case: {test_brackets}, Output: {output}")