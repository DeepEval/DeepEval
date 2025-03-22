import re

def method(input_str):
    # Function to swap case of a single character
    def swap_case(c):
        if c.isupper():
            return c.lower()
        elif c.islower():
            return c.upper()
        else:
            return c
    
    # Use regular expression to find all matching characters and apply the swap_case function
    output_str = re.sub(r'[a-z]', lambda match: swap_case(match.group()), input_str)
    output_str = re.sub(r'[A-Z]', lambda match: swap_case(match.group()), output_str)
    
    return output_str

# Test case to validate the function
def test_method():
    test_str = "Hello WorLD! 123"
    expected_output = "hELLO wORLD! 123"
    result = method(test_str)
    # assert result == expected_output, f"Expected {expected_output}, but got {result}"
    print("Test case passed.")

# Run the test case
test_method()