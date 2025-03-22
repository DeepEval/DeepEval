def method(text):
    """
    Returns encoded string by shifting every character by 5 in the alphabet.

    Args:
      text: The string to be encoded.

    Returns:
      The encoded string.
    """
    result = ''
    for char in text:
        if char.isalpha():
            start = ord('a') if char.islower() else ord('A')
            shifted_char = chr((ord(char) - start + 5) % 26 + start)
        else:
            shifted_char = char
        result += shifted_char
    return result

# Test Case
text = "Hello, World!"
encoded_text = method(text)
print(f"Original Text: {text}")
print(f"Encoded Text: {encoded_text}")