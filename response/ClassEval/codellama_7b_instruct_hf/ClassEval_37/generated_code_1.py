import string

class EncryptionUtils:
    def __init__(self, key):
        self.key = key
        self.ciphers = {"Caesar": self.caesar_cipher, "Vigenere": self.vigenere_cipher, "Rail Fence": self.rail_fence_cipher}

    def caesar_cipher(self, plaintext, shift):
        ciphertext = ""
        for char in plaintext:
            if char in string.ascii_letters:
                ciphertext += chr((ord(char) + shift - 97) % 26 + 97)
            else:
                ciphertext += char
        return ciphertext

    def vigenere_cipher(self, plaintext):
        ciphertext = ""
        for i, char in enumerate(plaintext):
            if char in string.ascii_letters:
                ciphertext += chr((ord(char) + ord(self.key[i % len(self.key)]) - 97) % 26 + 97)
            else:
                ciphertext += char
        return ciphertext

    def rail_fence_cipher(self, plaintext, rails):
        ciphertext = ""
        for char in plaintext:
            if char in string.ascii_letters:
                ciphertext += chr((ord(char) + (rails - 1) * 26) % 26 + 97)
            else:
                ciphertext += char
        return ciphertext

if __name__ == "__main__":
    e = EncryptionUtils("key")
    print(e.caesar_cipher("abc", 1))
    print(e.vigenere_cipher("abc"))
    print(e.rail_fence_cipher("abc", 2))