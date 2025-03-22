def method(text):
    import re
    
    # Replace more than two consecutive spaces with a single underscore
    text = re.sub(r'\s{3,}', '_', text)
    
    # Replace all remaining spaces with underscores
    output = text.replace(' ', '_')
    
    return output

# Test case for validation
test_text = "This  is    a   test    string with  spaces"
print(method(test_text))  # Expected output: "This__is_a_test_string_with_spaces"