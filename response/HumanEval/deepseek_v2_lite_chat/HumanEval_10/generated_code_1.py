def method(input_string):
    # Helper function to clean the input string
    def clean_string(s):
        return ''.join(c for c in s if c.isalnum()).lower()

    # Clean the input string
    cleaned_string = clean_string(input_string)

    # Check if the cleaned string is a palindrome
    is_palindrome = cleaned_string == cleaned_string[::-1]

    # Return the result
    return {'is_palindrome': is_palindrome, 'input_string': input_string}

# Example usage with a test case
result = method("A man, a plan, a canal: Panama")
print(result)