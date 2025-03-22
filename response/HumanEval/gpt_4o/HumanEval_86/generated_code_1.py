def method(input_string):
    # Split the input string into words and spaces
    words = input_string.split(' ')
    
    # Process each word to sort its characters
    sorted_words = [''.join(sorted(word)) for word in words]
    
    # Join the sorted words back with a single space
    output = ' '.join(sorted_words)
    
    return output

# Test case for validation
test_input = "hello world this is a test"
expected_output = "ehllo dlorw hist is a estt"
assert method(test_input) == expected_output, f"Test failed: {method(test_input)} != {expected_output}"
print("Test passed!")