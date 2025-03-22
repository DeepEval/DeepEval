def method(s, c):
    # Create a set of characters to remove for O(1) lookup
    chars_to_remove = set(c)
    
    # Create a result string by excluding characters in chars_to_remove
    result = ''.join([char for char in s if char not in chars_to_remove])
    
    # Check if the result string is a palindrome
    is_palindrome = result == result[::-1]
    
    # Return the result string and the palindrome check
    return (result, is_palindrome)

# Test case
test_s = "A man, a plan, a canal: Panama"
test_c = " ,:"
output = method(test_s, test_c)
print(output)  # Expected output: ('AmanaplanacanalPanama', True)