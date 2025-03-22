def method(message):
    def swap_case(c):
        return c.lower() if c.isupper() else c.upper()

    def replace_vowel(c):
        vowels = 'aeiou'
        if c in vowels:
            index = vowels.index(c)
            return vowels[(index + 2) % len(vowels)]
        return c

    output = []
    for char in message:
        char = swap_case(char)
        char = replace_vowel(char)
        output.append(char)

    return ''.join(output)

# Test case
input_message = "Hello World"
encoded_message = method(input_message)
print(f"Encoded message: {encoded_message}")  # Expected output: "Ifmmp Xpsme"