def method(input_string):
    encoded_string = []
    
    for char in input_string:
        # Check if the character is a lowercase letter
        if 'a' <= char <= 'z':
            # Shift character and wrap around using modulo
            shifted_char = chr((ord(char) - ord('a') + 5) % 26 + ord('a'))
            encoded_string.append(shifted_char)
        # Check if the character is an uppercase letter
        elif 'A' <= char <= 'Z':
            # Shift character and wrap around using modulo
            shifted_char = chr((ord(char) - ord('A') + 5) % 26 + ord('A'))
            encoded_string.append(shifted_char)
        else:
            # Non-alphabetical characters are not changed
            encoded_string.append(char)

    # Join the list into a single string
    output = ''.join(encoded_string)
    return output

# Test case
input_string = "Hello, World!"
output = method(input_string)
print(output)  # Expected output: "Mjqqt, Btwqi!"