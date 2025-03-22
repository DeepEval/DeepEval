def method(input_string):
    # Convert the string to lowercase to ensure case insensitivity
    lowercase_string = input_string.lower()
    
    # Use a set to find distinct characters
    distinct_characters = set(lowercase_string)
    
    # Calculate the number of distinct characters
    output = len(distinct_characters)
    
    return output

# Test case
test_input = "HelloWorld"
print(method(test_input))  # Should output 7: 'h', 'e', 'l', 'o', 'w', 'r', 'd'