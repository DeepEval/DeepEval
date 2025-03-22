def method(input_string):
    # Remove any spaces and convert the string to lowercase for a case-insensitive comparison
    processed_string = ''.join(input_string.split()).lower()
    
    # Check if the processed string is equal to its reverse
    is_palindrome = processed_string == processed_string[::-1]
    
    return is_palindrome

# Test case for validation
test_string = "A man a plan a canal Panama"
print(method(test_string))  # Expected output: True