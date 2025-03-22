def method(message):
    # Define a mapping of vowels to their replacements
    vowel_replacement = {
        'a': 'c',
        'e': 'g',
        'i': 'k',
        'o': 'q',
        'u': 'w',
        'A': 'C',
        'E': 'G',
        'I': 'K',
        'O': 'Q',
        'U': 'W'
    }
    
    # Initialize an empty list to store the transformed characters
    transformed_message = []
    
    # Iterate over each character in the message
    for char in message:
        if char in vowel_replacement:
            # Replace the vowel with its corresponding letter
            transformed_message.append(vowel_replacement[char])
        else:
            # Swap the case of the letter
            transformed_message.append(char.swapcase())
    
    # Join the list of characters into a single string
    output = ''.join(transformed_message)
    
    return output

# Test case for validation
test_message = "Hello World"
expected_output = "hGllQ wQrlD"

# Run the method with the test message
print("Test Output: ", method(test_message))
print("Expected Output: ", expected_output)