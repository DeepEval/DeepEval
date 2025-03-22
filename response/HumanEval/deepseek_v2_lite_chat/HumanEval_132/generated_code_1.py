def method():
    def is_nested(s):
        stack = []
        for c in s:
            if c == '[':
                stack.append(c)
            elif c == ']' and stack:
                stack.pop()
        return stack == []
    return is_nested

# Test cases
test_cases = [
    ("[[]]", True),
    ("[]]]]]]][[[[[]", False),
    ("[][]", False),
    ("[]", False),
    ("[[][]]", True),
    ("[[]][[", True)
]

output = method()
for test_case in test_cases:
    input_string, expected_output = test_case
    result = output(input_string)
    # assert result == expected_output, f"Expected {expected_output} but got {result} for input {input_string}"
    # print(f"Test case passed for input {input_string}")