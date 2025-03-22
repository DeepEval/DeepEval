def method():
    # Example input
    input_string = "racecar"
    
    # Function to check if a string is a palindrome
    def is_palindrome(s):
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned_string = ''.join(char.lower() for char in s if char.isalnum())
        # Compare the cleaned string with its reverse
        return cleaned_string == cleaned_string[::-1]
    
    # Check if the input string is a palindrome
    output = is_palindrome(input_string)
    
    return output

# Test case
print(method())  # Expected output: True