def method(message):
    output = []
    for char in message:
        if char.isalpha():
            if char in 'aeiouAEIOU':
                shifted_char = chr((ord(char) - ord('a' if char.islower() else 'A') + 2) % 26 + ord('a' if char.islower() else 'A'))
            else:
                shifted_char = char
            output.append(shifted_char.swapcase())
        else:
            output.append(char)
    return ''.join(output)

# Test case
message = "Hello, World!"
encoded_message = method(message)
print(f"Original message: {message}")
print(f"Encoded message: {encoded_message}")