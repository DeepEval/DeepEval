def encrypt(text):
    def shift_char(c, shift_amount):
        if 'a' <= c <= 'z':
            return chr((ord(c) - ord('a') + shift_amount) % 26 + ord('a'))
        elif 'A' <= c <= 'Z':
            return chr((ord(c) - ord('A') + shift_amount) % 26 + ord('A'))
        else:
            return c

    shift_amount = 4  # Two multiplied by two
    encrypted_text = ''.join(shift_char(c, shift_amount) for c in text)
    
    return encrypted_text

def method():
    # Example test case
    input_text = "Hello, World!"
    output = encrypt(input_text)
    return output

# Test the function
print(method())  # Expected output: "Lipps, Asvph!"