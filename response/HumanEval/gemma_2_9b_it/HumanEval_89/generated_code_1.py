def method():
    def encrypt(text):
        result = ''
        for char in text:
            if char.isalpha():
                start = ord('a') if char.islower() else ord('A')
                shifted_char = chr((ord(char) - start + 8) % 26 + start)
            else:
                shifted_char = char
            result += shifted_char
        return result
    
    text = "Hello, World!"
    encrypted_text = encrypt(text)
    
    output = f"Original text: {text}\nEncrypted text: {encrypted_text}"
    return output

print(method())