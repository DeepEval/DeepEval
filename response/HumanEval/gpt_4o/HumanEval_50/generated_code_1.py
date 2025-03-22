def method(input_string):
    output = []
    
    for char in input_string:
        if char.isalpha():  # Check if the character is an alphabet
            shift = 5
            if char.islower():
                # Calculate the new position with wrapping
                new_char = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            else:
                # For uppercase letters
                new_char = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            output.append(new_char)
        else:
            output.append(char)  # Non-alphabetic characters remain unchanged
    
    return ''.join(output)

# Test case for validation
test_input = "Hello, World!"
expected_output = "Mjqqt, Btwqi!"
actual_output = method(test_input)

print(f"Test Input: {test_input}")
print(f"Expected Output: {expected_output}")
print(f"Actual Output: {actual_output}")
assert actual_output == expected_output, "Test case failed!"
print("Test case passed!")