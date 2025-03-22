from string import ascii_lowercase, ascii_uppercase

def encrypt(text):
    """
    Encrypts a string by rotating the alphabet by two multiplied to two places.

    Args:
        text: The string to encrypt.

    Returns:
        The encrypted string.
    """

    encrypted_text = ""

    for char in text:
        if char.lower() in ascii_lowercase:
            # Find the index of the character in the alphabet.
            index = ascii_lowercase.index(char.lower())

            # Rotate the character by two multiplied to two places.
            new_index = (index + 4) % 26

            # Add the encrypted character to the output string.
            encrypted_text += ascii_lowercase[new_index]
        elif char.upper() in ascii_uppercase:
            # Find the index of the character in the alphabet.
            index = ascii_uppercase.index(char)

            # Rotate the character by two multiplied to two places.
            new_index = (index + 4) % 26

            # Add the encrypted character to the output string.
            encrypted_text += ascii_uppercase[new_index]
        else:
            # Add the character to the output string as is.
            encrypted_text += char

    return encrypted_text


# Test case
text = "Hello, world!"
encrypted_text = encrypt(text)

print(encrypted_text)