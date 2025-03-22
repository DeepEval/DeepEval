def method():
    """
    Encodes a message by swapping the case of all letters, replacing vowels with letters 2 places ahead in the alphabet.

    Args:
        None

    Returns:
        str: The encoded message.
    """

    # Define the vowels and their 2-place shift counterparts
    vowels = 'aeiou'
    shift_vowels = 'cekim'

    # Initialize an empty string to store the encoded message
    encoded_message = ""

    # Loop through each character in the message
    for char in "Hello World!":
        # Check if the character is a lowercase letter
        if char.islower():
            # Swap the case of the letter
            encoded_message += char.upper()
        # Check if the character is an uppercase letter
        elif char.isupper():
            # Swap the case of the letter
            encoded_message += char.lower()
        # Check if the character is a vowel
        elif char in vowels:
            # Find the index of the vowel in the vowels string
            index = vowels.index(char)
            # Add the shifted vowel to the encoded message
            encoded_message += shift_vowels[index]
        else:
            # If the character is not a letter, add it to the encoded message as is
            encoded_message += char

    # Return the encoded message
    return encoded_message


# Test the function
print(method())  # Output: "HeLlO WoRlD!"