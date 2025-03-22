import re

def method(input_string):
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string).lower()
    
    # Check if the cleaned string is equal to its reverse
    is_palindrome = cleaned_string == cleaned_string[::-1]
    
    return is_palindrome

# Test case for validation
test_string = "A man, a plan, a canal: Panama"
print(f"Is the test string a palindrome? {method(test_string)}")  # Expected output: True