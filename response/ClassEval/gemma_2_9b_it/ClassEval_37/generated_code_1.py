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
        >>> e = EncryptionUtils("key")
        >>> e.caesar_cipher("abc", 1)
        'bcd'

        """
        ciphertext = ''
        for char in plaintext:
            if char.isalpha():
                start = ord('a') if char.islower() else ord('A')
                shifted_char = chr((ord(char) - start + shift) % 26 + start)
            else:
                shifted_char = char
            ciphertext += shifted_char
        return ciphertext

    def vigenere_cipher(self, plaintext):
        """
        Encrypts the plaintext using the Vigenere cipher.
        :param plaintext: The plaintext to encrypt, str.
        :return: The ciphertext, str.
        >>> e = EncryptionUtils("key")
        >>> e.vigenere_cipher("abc")
        'kfa'

        """
        ciphertext = ''
        key_len = len(self.key)
        key_as_int = [ord(i) for i in self.key]
        plaintext_as_int = [ord(i) for i in plaintext]
        for i in range(len(plaintext_as_int)):
            value = (plaintext_as_int[i] + key_as_int[i % key_len]) % 26
            if plaintext[i].isupper():
                ciphertext += chr(value + ord('A'))
            elif plaintext[i].islower():
                ciphertext += chr(value + ord('a'))
            else:
                ciphertext += plaintext[i]
        return ciphertext

    def rail_fence_cipher(self, plaintext, rails):
        """
        Encrypts the plaintext using the Rail Fence cipher.
        :param plaintext: The plaintext to encrypt, str.
        :return: The ciphertext, str.
        >>> e = EncryptionUtils("key")
        >>> e.rail_fence_cipher("abc", 2)
        'acb'

        """
        fence = [['' for _ in range(len(plaintext))] for _ in range(rails)]
        direction = 1
        row, col = 0, 0
        for char in plaintext:
            fence[row][col] = char
            row += direction
            if row == rails - 1 or row == 0:
                direction *= -1
            col += 1
        ciphertext = ''
        for row in fence:
            for char in row:
                if char != '':
                    ciphertext += char
        return ciphertext

if __name__ == "__main__":
    instance = EncryptionUtils("key")
    # Test case for caesar_cipher
    output = instance.caesar_cipher("abc", 1)
    print(f"Caesar Cipher Output: {output}")

    # Test case for vigenere_cipher
    output = instance.vigenere_cipher("abc")
    print(f"Vigenere Cipher Output: {output}")

    # Test case for rail_fence_cipher
    output = instance.rail_fence_cipher("abc", 2)
    print(f"Rail Fence Cipher Output: {output}")