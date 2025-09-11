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
        output = ""
        for char in plaintext:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                output += chr((ord(char) - base + shift) % 26 + base)
            else:
                output += char
        return output

    def vigenere_cipher(self, plaintext):
        """
        Encrypts the plaintext using the Vigenere cipher.
        :param plaintext: The plaintext to encrypt, str.
        :return: The ciphertext, str.
        >>> e = EncryptionUtils("key")
        >>> e.vigenere_cipher("abc")
        'kfa'

        """
        output = ""
        key_index = 0
        for char in plaintext:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                key_char = self.key[key_index % len(self.key)]
                shift = ord(key_char) - base
                output += chr((ord(char) - base + shift) % 26 + base)
                key_index += 1
            else:
                output += char
        return output

    def rail_fence_cipher(self, plain_text, rails):
        """
        Encrypts the plaintext using the Rail Fence cipher.
        :param plaintext: The plaintext to encrypt, str.
        :return: The ciphertext, str.
        >>> e = EncryptionUtils("key")
        >>> e.rail_fence_cipher("abc", 2)
        'acb'

        """
        output = ["" for _ in range(rails)]
        row = 0
        direction = "down"
        for char in plain_text:
            output[row] += char
            if row == 0 or row == rails - 1:
                direction = "up" if direction == "down" else "down"
            row += 1 if direction == "down" else -1
        return "".join(output)


if __name__ == "__main__":
    e = EncryptionUtils("key")
    # Test case for caesar_cipher
    output = e.caesar_cipher("abc", 1)
    print(output)  # Output: bcd

    # Test case for vigenere_cipher
    output = e.vigenere_cipher("abc")
    print(output)  # Output: kfa

    # Test case for rail_fence_cipher
    output = e.rail_fence_cipher("abc", 2)
    print(output)  # Output: acb