def method():
    def is_nested(s):
        stack = []
        for bracket in s:
            if bracket == '[':
                stack.append(bracket)
            elif bracket == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                else:
                    return False
        return len(stack) == 0

    # Test cases
    test_cases = ['[[]]', '[]]]]]]][[[[[]', '[][]', '[]', '[[][]]', '[[]][[']
    results = [is_nested(test) for test in test_cases]
    
    output = all(results)
    return output

# Run the method and print the result
print(method())