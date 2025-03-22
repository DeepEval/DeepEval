def method(s):
    # Normalize the string by removing spaces and converting to lowercase
    normalized_string = ''.join(s.split()).lower()
    
    # Check if the normalized string is equal to its reverse
    is_palindrome = normalized_string == normalized_string[::-1]
    
    return is_palindrome

# Test case
test_string = "A man a plan a canal Panama"
output = method(test_string)
print(f"Is the string '{test_string}' a palindrome? {output}")