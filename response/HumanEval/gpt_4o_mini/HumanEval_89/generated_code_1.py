def encrypt(text):
    encrypted_text = []
    shift = 4  # As per the requirement, shift down by 2 * 2

    for char in text:
        if char.isalpha():
            if char.islower():
                # Rotate lowercase letters
                new_char = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            else:
                # Rotate uppercase letters
                new_char = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            encrypted_text.append(new_char)
        else:
            # Non-alphabetical characters remain unchanged
            encrypted_text.append(char)

    return ''.join(encrypted_text)

def method():
    # Test case for validation
    test_string = "Hello, World!"
    output = encrypt(test_string)
    return output

# Running the method to see the output
if __name__ == "__main__":
    result = method()
    print(result)