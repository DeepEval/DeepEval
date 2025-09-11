class EncryptionUtils:
    """
    This class provides methods for encryption using the Caesar cipher, Vigenere cipher, and Rail Fence cipher.
    """

    def __init__(self, key):
        """
        Initializes the class with a key.
        :param key: The key to use for encryption, str.
        """
        self.key = key

    def caesar_cipher(self, plaintext, shift):
        """
        Encrypts the plaintext using the Caesar cipher.
        :param plaintext: The plaintext to encrypt, str.
        :param shift: The number of characters to shift each character in the plaintext, int.
        :return: The ciphertext, str.
        """
        ciphertext = []
        for char in plaintext:
            if char.isalpha():
                ascii_offset = ord('a') if char.islower() else ord('A')
                ciphered_char = chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
                ciphertext.append(ciphered_char)
            else:
                ciphertext.append(char)
        return ''.join(ciphertext)

    def vigenere_cipher(self, plaintext):
        """
        Encrypts the plaintext using the Vigenere cipher.
        :param plaintext: The plaintext to encrypt, str.
        :return: The ciphertext, str.
        """
        key_length = len(self.key)
        if key_length == 0:
            raise ValueError("Key must not be empty for Vigenere cipher")

        ciphertext = []
        for i in range(len(plaintext)):
            key_char_index = i % key_length
            key_char = self.key[key_char_index]
            plain_char = plaintext[i]
            if plain_char.isalpha():
                ascii_offset = ord('a') if plain_char.islower() else ord('A')
                ciphered_char = chr((ord(plain_char) - ascii_offset + (ord(key_char) - ord('a')) * (i // key_length)) % 26 + ascii_offset)
                ciphertext.append(ciphered_char)
            else:
                ciphertext.append(plain_char)
        return ''.join(ciphertext)

    def rail_fence_cipher(self, plain_text, rails):
        """
        Encrypts the plaintext using the Rail Fence cipher.
        :param plaintext: The plaintext to encrypt, str.
        :param rails: The number of rails in the Rail Fence cipher, int.
        :return: The ciphertext, str.
        """
        if rails < 1 or not plain_text.isalpha():
            raise ValueError("Number of rails must be positive and plaintext must be alphabetic")

        ciphertext = plain_text.replace(" ", "")  # Remove spaces first
        rows = [plain_text[i:i+rails] for i in range(rails)]
        ciphered_text = [''] * rails
        for i in range(len(ciphertext)):
            ciphered_text[(i * 2) % rails].append(rows[i][(i * 2) % rails].rjust((i * 2) % rails, ' '))
            ciphered_text[(i * 2 + 1) % rails].append(ciphertext[i])
        return ''.join([''.join(row) for row in ciphered_text])


# Test cases
if __name__ == "__main__":
    e = EncryptionUtils("key")
    print(e.caesar_cipher("abc", 1))  # Should return 'bcd'
    print(e.vigenere_cipher("abc"))  # Should return 'kfa'
    print(e.rail_fence_cipher("abc", 2))  # Should return 'acb'