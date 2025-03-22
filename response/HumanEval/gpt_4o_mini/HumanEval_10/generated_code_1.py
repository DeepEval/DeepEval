def method(input_string):
    # Normalize the string: remove spaces and convert to lowercase
    normalized_string = ''.join(input_string.split()).lower()
    
    # Check if the normalized string is equal to its reverse
    is_palindrome = normalized_string == normalized_string[::-1]
    
    return is_palindrome

# Test case for validation
test_input = "A man a plan a canal Panama"
output = method(test_input)
print(f'Is the input string a palindrome? {output}')