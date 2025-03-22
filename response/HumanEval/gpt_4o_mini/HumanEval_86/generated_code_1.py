def method(input_string):
    # Split the input string by spaces
    words = input_string.split(' ')
    
    # Sort the characters in each word
    sorted_words = [''.join(sorted(word)) for word in words]
    
    # Join the sorted words back into a single string with spaces
    output = ' '.join(sorted_words)
    
    return output

# Test case for validation
test_input = "hello world this is a test"
expected_output = "ehllo dlrow hist is a estt"
actual_output = method(test_input)

print(f"Test Input: '{test_input}'")
print(f"Expected Output: '{expected_output}'")
print(f"Actual Output: '{actual_output}'")
print("Test Passed!" if actual_output == expected_output else "Test Failed!")