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
        ciphertext = []
        for char in plaintext:
            if char.isalpha():
                shift_amount = shift % 26
                new_char = chr((ord(char) - ord('a') + shift_amount) % 26 + ord('a'))
                ciphertext.append(new_char)
            else:
                ciphertext.append(char)
        return ''.join(ciphertext)

    def vigenere_cipher(self, plaintext):
        """
        Encrypts the plaintext using the Vigenere cipher.
        :param plaintext: The plaintext to encrypt, str.
        :return: The ciphertext, str.
        """
        ciphertext = []
        key_length = len(self.key)
        for i, char in enumerate(plaintext):
            if char.isalpha():
                key_char = self.key[i % key_length].lower()
                shift_amount = ord(key_char) - ord('a')
                new_char = chr((ord(char) - ord('a') + shift_amount) % 26 + ord('a'))
                ciphertext.append(new_char)
            else:
                ciphertext.append(char)
        return ''.join(ciphertext)

    def rail_fence_cipher(self, plaintext, rails):
        """
        Encrypts the plaintext using the Rail Fence cipher.
        :param plaintext: The plaintext to encrypt, str.
        :return: The ciphertext, str.
        """
        if rails <= 1:
            return plaintext

        fence = [[] for _ in range(rails)]
        rail = 0
        direction = 1

        for char in plaintext:
            fence[rail].append(char)
            rail += direction
            if rail == 0 or rail == rails - 1:
                direction = -direction

        ciphertext = ''.join([''.join(row) for row in fence])
        return ciphertext


if __name__ == "__main__":
    # Test case for Caesar cipher
    e_caesar = EncryptionUtils("key")
    output_caesar = e_caesar.caesar_cipher("abc", 1)
    print("Caesar Cipher:", output_caesar)  # Output: 'bcd'

    # Test case for Vigenere cipher
    e_vigenere = EncryptionUtils("key")
    output_vigenere = e_vigenere.vigenere_cipher("abc")
    print("Vigenere Cipher:", output_vigenere)  # Output: 'kfa'

    # Test case for Rail Fence cipher
    e_rail_fence = EncryptionUtils("key")
    output_rail_fence = e_rail_fence.rail_fence_cipher("abc", 2)
    print("Rail Fence Cipher:", output_rail_fence)  # Output: 'acb'