class EncryptionUtils:
    """
    This is a class that provides methods for encryption, including the Caesar cipher, Vigenere cipher, and Rail Fence cipher.
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
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        shifted_text = ""
        for char in plaintext:
            if char.isalpha():
                if char.islower():
                    shifted_text += alphabet[(alphabet.index(char) + shift) % 26]
                else:
                    shifted_text += alphabet[(alphabet.index(char.lower()) + shift) % 26].upper()
            else:
                shifted_text += char
        return shifted_text

    def vigenere_cipher(self, plaintext):
        """
        Encrypts the plaintext using the Vigenere cipher.
        :param plaintext: The plaintext to encrypt, str.
        :return: The ciphertext, str.
        """
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        key_index = 0
        shifted_text = ""
        for char in plaintext:
            if char.isalpha():
                shift = self.key[key_index % len(self.key)].lower()
                if char.islower():
                    shifted_text += alphabet[(alphabet.index(char) + alphabet.index(shift)) % 26]
                else:
                    shifted_text += alphabet[(alphabet.index(char.lower()) + alphabet.index(shift)) % 26].upper()
                key_index += 1
            else:
                shifted_text += char
        return shifted_text

    def rail_fence_cipher(self, plaintext, rails):
        """
        Encrypts the plaintext using the Rail Fence cipher.
        :param plaintext: The plaintext to encrypt, str.
        :return: The ciphertext, str.
        """
        rows = [''] * rails
        direction = 1
        row_index = 0
        for char in plaintext:
            rows[row_index] += char
            row_index += direction
            if row_index == 0 or row_index == rails - 1:
                direction *= -1
        ciphertext = ''.join(rows)
        return ciphertext

if __name__ == "__main__":
    e = EncryptionUtils("key")
    # Test case for Caesar Cipher
    output = e.caesar_cipher("abc", 1)
    print(f"Caesar Cipher Output: {output}")
    
    # Test case for Vigenere Cipher
    output = e.vigenere_cipher("abc")
    print(f"Vigenere Cipher Output: {output}")
    
    # Test case for Rail Fence Cipher
    output = e.rail_fence_cipher("abc", 2)
    print(f"Rail Fence Cipher Output: {output}")