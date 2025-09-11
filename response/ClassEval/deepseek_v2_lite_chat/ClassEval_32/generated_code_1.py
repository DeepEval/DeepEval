import re

class DecryptionUtils:
    def __init__(self, key):
        self.key = key

    def caesar_decipher(self, ciphertext, shift):
        plaintext = ""
        for char in ciphertext:
            if char.isalpha():
                ascii_offset = ord('a') if char.islower() else ord('A')
                shifted_char = chr((ord(char) - shift - ascii_offset) % 26 + ascii_offset)
                plaintext += shifted_char
            else:
                plaintext += char
        return plaintext

    def vigenere_decipher(self, ciphertext):
        plaintext = ""
        key_length = len(self.key)
        for i, char in enumerate(ciphertext):
            if char.isalpha():
                key_index = i % key_length
                key_char = self.key[key_index]
                ascii_offset = ord('a') if char.islower() else ord('A')
                shifted_char = chr((ord(char) - ord(key_char) - shift + 26) % 26 + ascii_offset)
                plaintext += shifted_char
            else:
                plaintext += char
        return plaintext

    def rail_fence_decipher(self, encrypted_text, rails):
        plaintext = ""
        rail_pattern = re.compile(r'..(?!.)*?[^\S\n]{%d,}(?!.)\1*' % rails)
        rail_map = range(rails)
        for layer in rail_pattern.findall(encrypted_text):
            plaintext += ''.join(encrypted_text[i+rails] for i in rail_map)
        return plaintext

# Test cases
def test_decryption_utils():
    d = DecryptionUtils('key')

    # Test Caesar cipher
    assert d.caesar_decipher('ifmmp', 1) == 'hello'
    assert d.caesar_decipher('ABCDEF', 1) == 'ABCDEF'
    assert d.caesar_decipher('hijkl', 1) == 'hijkl'

    # Test Vigenere cipher
    assert d.vigenere_decipher('ifmmp') == 'ybocl'
    assert d.vigenere_decipher('ABCDEF') == 'ABCDEF'
    assert d.vigenere_decipher('hijkl') == 'hijkl'

    # Test Rail Fence cipher
    assert d.rail_fence_decipher('Hoo!el,Wrdl l', 3) == 'Hello, World!'
    assert d.rail_fence_decipher('ifmmp') == 'ybocl'
    assert d.rail_fence_decipher('abcdefgh') == 'abcdefgh'

if __name__ == "__main__":
    test_decryption_utils()
    print("All test cases pass")