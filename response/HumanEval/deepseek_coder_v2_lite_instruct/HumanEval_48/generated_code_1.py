def method():
    # Input string
    input_string = "racecar"
    
    # Function to check if the string is a palindrome
    def is_palindrome(s):
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned_string = ''.join(char.lower() for char in s if char.isalnum())
        # Check if the cleaned string is equal to its reverse
        return cleaned_string == cleaned_string[::-1]
    
    # Call the function and store the result
    output = is_palindrome(input_string)
    
    return output

# # Test case for validation
# def test_method():
#     assert method() == True, "Test case failed: 'racecar' should be a palindrome"
#     assert method("hello") == False, "Test case failed: 'hello' should not be a palindrome"
#     print("All test cases passed!")

# # Run the test case
# test_method()

# Run the method
method()