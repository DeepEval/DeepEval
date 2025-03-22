def method(s, c):
    # Remove characters from s that are present in c
    result_string = ''.join([char for char in s if char not in c])
    
    # Check if the result string is a palindrome
    is_palindrome = result_string == result_string[::-1]
    
    # Return the result string and whether it is a palindrome
    return (result_string, is_palindrome)

# Test case
s = "hello world"
c = "o"
output = method(s, c)
print(output)  # Expected output: ('hell wrld', False)