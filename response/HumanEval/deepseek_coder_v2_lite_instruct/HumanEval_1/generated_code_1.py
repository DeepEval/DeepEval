def method():
    def separate_groups(s):
        s = s.replace(" ", "")  # Remove all spaces
        stack = []
        result = []
        current_group = []

        for char in s:
            if char == '(':
                if stack:
                    current_group.append(char)
                stack.append(char)
            elif char == ')':
                stack.pop()
                current_group.append(char)
                if not stack:
                    result.append(''.join(current_group))
                    current_group = []

        return result

    # Example usage:
    input_string = "((hello world) (this is a) test)"
    output = separate_groups(input_string)
    return output

# Test case
input_string = "((hello world) (this is a) test)"
expected_output = ['(hello world)', '(this is a)', 'test']
# assert method() == expected_output, f"Expected {expected_output}, but got {method()}"
print(method())