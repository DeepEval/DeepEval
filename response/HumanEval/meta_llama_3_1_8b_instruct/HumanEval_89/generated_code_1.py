def encrypt(text):
    """
    Encrypts the input string by shifting the alphabet down by 4 places.
    
    Args:
    text (str): The input string to be encrypted.
    
    Returns:
    str: The encrypted string.
    """
    
    # Define the lowercase and uppercase alphabets
    lowercase_alphabet = 'abcdefghijklmnopqrstuvwxyz'
    uppercase_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    # Initialize an empty string to store the encrypted text
    encrypted_text = ''
    
    # Iterate over each character in the input text
    for char in text:
        
        # Check if the character is in the lowercase alphabet
        if char in lowercase_alphabet:
            # Find the index of the character in the lowercase alphabet
            index = lowercase_alphabet.index(char)
            # Shift the index down by 4 places and use the modulus operator to wrap around the alphabet
            encrypted_index = (index - 4) % 26
            # Append the encrypted character to the encrypted text
            encrypted_text += lowercase_alphabet[encrypted_index]
        
        # Check if the character is in the uppercase alphabet
        elif char in uppercase_alphabet:
            # Find the index of the character in the uppercase alphabet
            index = uppercase_alphabet.index(char)
            # Shift the index down by 4 places and use the modulus operator to wrap around the alphabet
            encrypted_index = (index - 4) % 26
            # Append the encrypted character to the encrypted text
            encrypted_text += uppercase_alphabet[encrypted_index]
        
        # If the character is not in the alphabet, append it to the encrypted text as is
        else:
            encrypted_text += char
    
    # Return the encrypted text
    return encrypted_text


# Test the function
test_text = "Hello, World!"
encrypted_text = encrypt(test_text)
print(f"Input: {test_text}")
print(f"Encrypted: {encrypted_text}")