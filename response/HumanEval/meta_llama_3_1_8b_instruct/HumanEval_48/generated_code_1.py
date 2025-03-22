def method():
    """
    Checks if given string is a palindrome.
    
    Args:
    input_str (str): The input string to check.
    
    Returns:
    bool: True if the string is a palindrome, False otherwise.
    """

    # Remove spaces and convert to lowercase
    input_str = ''.join(e for e in input_str if e.isalnum()).lower()
    
    # Compare the string with its reverse
    return input_str == input_str[::-1]


# Test case
input_str = "A man, a plan, a canal: Panama"
output = method()
print(f"Is '{input_str}' a palindrome? {output}")