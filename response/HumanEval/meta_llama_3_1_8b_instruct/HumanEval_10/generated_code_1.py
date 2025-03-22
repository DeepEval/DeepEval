def is_palindrome(s: str) -> bool:
    """
    Checks if the given string is a palindrome.
    
    Args:
    s (str): The input string to check.
    
    Returns:
    bool: True if the string is a palindrome, False otherwise.
    """
    # Remove any non-alphanumeric characters and convert to lowercase
    s = ''.join(e for e in s if e.isalnum()).lower()
    
    # Compare the string with its reverse
    return s == s[::-1]


def method():
    # Test case
    input_string = "A man, a plan, a canal: Panama"
    output = is_palindrome(input_string)
    
    return output


# Test case
print(method())