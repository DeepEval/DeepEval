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
        ciphertext = ""
        for char in plaintext:
            if char.isalpha():
                shift_amount = shift % 26
                if char.islower():
                    ciphertext += chr((ord(char) - ord('a') + shift_amount) % 26 + ord('a'))
                else:
                    ciphertext += chr((ord(char) - ord('A') + shift_amount) % 26 + ord('A'))
            else:
                ciphertext += char
        return ciphertext

    def vigenere_cipher(self, plaintext):
        """
        Encrypts the plaintext using the Vigenere cipher.
        :param plaintext: The plaintext to encrypt, str.
        :return: The ciphertext, str.
        """
        ciphertext = ""
        key_length = len(self.key)
        for i, char in enumerate(plaintext):
            if char.isalpha():
                key_char = self.key[i % key_length]
                shift = ord(key_char.upper()) - ord('A')
                if char.islower():
                    ciphertext += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
                else:
                    ciphertext += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            else:
                ciphertext += char
        return ciphertext

    def rail_fence_cipher(self, plaintext, rails):
        """
        Encrypts the plaintext using the Rail Fence cipher.
        :param plaintext: The plaintext to encrypt, str.
        :param rails: The number of rails to use for encryption, int.
        :return: The ciphertext, str.
        """
        if rails == 1:
            return plaintext
        rail_pattern = [[] for _ in range(rails)]
        rail = 0
        direction = 1
        for char in plaintext:
            rail_pattern[rail].append(char)
            rail += direction
            if rail == 0 or rail == rails - 1:
                direction *= -1
        ciphertext = ''.join([''.join(rail) for rail in rail_pattern])
        return ciphertext

# Test cases
if __name__ == "__main__":
    e = EncryptionUtils("key")
    
    # Test Caesar Cipher
    print(e.caesar_cipher("abc", 1))  # Output: 'bcd'
    
    # Test Vigenere Cipher
    print(e.vigenere_cipher("abc"))  # Output: 'kfa'
    
    # Test Rail Fence Cipher
    print(e.rail_fence_cipher("abc", 2))  # Output: 'acb'