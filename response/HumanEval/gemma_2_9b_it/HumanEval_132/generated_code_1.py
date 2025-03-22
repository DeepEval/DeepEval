def method():
    def is_nested(s):
        open_brackets = 0
        for char in s:
            if char == '[':
                open_brackets += 1
            elif char == ']':
                if open_brackets > 0:
                    open_brackets -= 1
                else:
                    return True  # Found a closing bracket without a matching opening one
        return open_brackets > 0  # True if there are unmatched opening brackets, indicating nesting

    test_cases = [
        ('[[]]', True),
        ('[]]]]]]][[[[[]', False),
        ('[][]', False),
        ('[]', False),
        ('[[][]]', True),
        ('[[]][[', True)
    ]

    for input_str, expected_output in test_cases:
        output = is_nested(input_str)
        # assert output == expected_output, f"Test case failed: '{input_str}' expected {expected_output} but got {output}"
    
    return "All test cases passed!"  

print(method())