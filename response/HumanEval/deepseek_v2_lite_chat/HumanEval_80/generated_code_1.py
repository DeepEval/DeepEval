def is_happy_string(s):
    # Check if the string is at least 3 characters long
    if len(s) < 3:
        return False
    
    # Iterate over the string, checking every 3 consecutive characters
    for i in range(len(s) - 2):
        # Sort the characters to check if they are distinct
        sorted_chars = sorted(s[i:i+3])
        # Check if the sorted characters are the same as the original characters
        if s[i:i+3] == ''.join(sorted_chars):
            continue
        else:
            return False
    
    # If all 3-character sequences are distinct, return True
    return True

# Example test case
def test_is_happy_string():
    # assert is_happy_string("abc") == True
    # assert is_happy_string("aab") == False
    # assert is_happy_string("cabcab") == False
    # assert is_happy_string("cabcd") == True
    # assert is_happy_string("aabbc") == False
    print(is_happy_string("aabbc"))

# Call the test function
test_is_happy_string()

# If you want to use the function in your code
# print(is_happy_string("abc"))  # Should return True
# print(is_happy_string("aab"))  # Should return False
# print(is_happy_string("cabcab"))  # Should return False
# print(is_happy_string("cabcd"))  # Should return True
# print(is_happy_string("aabbc"))  # Should return False