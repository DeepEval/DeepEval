def method(input_string):
    # Check for whitespace in the string
    if ' ' in input_string:
        # Split on whitespace and return the list of words
        output = input_string.split()
    elif ',' in input_string:
        # Split on commas if no whitespace exists
        output = input_string.split(',')
    else:
        # If there are no whitespaces or commas, count the lowercase letters with odd orders
        count = sum(1 for char in input_string if char.islower() and (ord(char) - ord('a')) % 2 == 1)
        output = count
    
    return output

# Test case for validation
test_string_1 = "Hello world this is a test"  # Should split into words
test_string_2 = "apple,banana,cherry"          # Should split on commas
test_string_3 = "abcdefg"                       # Should count lowercase letters with odd order

print(method(test_string_1))  # Output: ['Hello', 'world', 'this', 'is', 'a', 'test']
print(method(test_string_2))  # Output: ['apple', 'banana', 'cherry']
print(method(test_string_3))  # Output: 3 (letters 'b', 'd', 'f' are in odd positions)