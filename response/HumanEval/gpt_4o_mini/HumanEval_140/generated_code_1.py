import re

def method(text):
    # Replace more than 2 consecutive spaces with a single space
    text = re.sub(r' {3,}', ' ', text)
    
    # Replace all remaining spaces with underscores
    text = text.replace(' ', '_')
    
    return text

# Test case for validation
if __name__ == "__main__":
    test_input = "This  is   a test    string with  multiple spaces."
    expected_output = "This_is_a_test_string_with_multiple_spaces."
    
    result = method(test_input)
    print(f"Input: {test_input}")
    print(f"Output: {result}")
    print(f"Expected: {expected_output}")
    print(f"Test Passed: {result == expected_output}")