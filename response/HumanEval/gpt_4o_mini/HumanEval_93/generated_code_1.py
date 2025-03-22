def method(message):
    # Define a dictionary to map vowels to their corresponding replacements
    vowel_replacements = {
        'a': 'c', 'A': 'C',
        'e': 'g', 'E': 'G',
        'i': 'k', 'I': 'K',
        'o': 'q', 'O': 'Q',
        'u': 'w', 'U': 'W'
    }
    
    # Initialize an empty list to hold the transformed characters
    transformed_message = []

    # Iterate over each character in the message
    for char in message:
        # Swap case
        if char.isalpha():
            swapped_char = char.swapcase()
            # Replace vowel if it's a vowel
            if swapped_char in vowel_replacements:
                transformed_message.append(vowel_replacements[swapped_char])
            else:
                transformed_message.append(swapped_char)
        else:
            # If it's not an alphabetic character, keep it as is
            transformed_message.append(char)

    # Join the transformed characters into a single string
    output = ''.join(transformed_message)
    return output

# Test case for validation
test_message = "Hello World! This is an example."
output = method(test_message)
print(output)  # Expected output: "gqMMO wqRLD! tHIS kG cW cKqMCg."