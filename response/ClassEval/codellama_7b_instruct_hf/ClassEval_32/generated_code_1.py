import string

class DecryptionUtils:
    """
    This is a class that provides methods for decryption, including the Caesar cipher, Vigenere cipher, and Rail Fence cipher.
    """

    def __init__(self, key):
        """
        Initializes the decryption utility with a key.
        :param key: The key to use for decryption,str.
        """
        self.key = key

    def caesar_decipher(self, ciphertext, shift):
        """
        Deciphers the given ciphertext using the Caesar cipher
        :param ciphertext: The ciphertext to decipher,str.
        :param shift: The shift to use for decryption,int.
        :return: The deciphered plaintext,str.
        >>> d = DecryptionUtils('key')
        >>> d.caesar_decipher('ifmmp', 1)
        'hello'

        """
        plaintext = ""
        for char in ciphertext:
            if char in string.ascii_letters:
                plaintext += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            else:
                plaintext += char
        return plaintext

    def vigenere_decipher(self, ciphertext):
        """
        Deciphers the given ciphertext using the Vigenere cipher
        :param ciphertext: The ciphertext to decipher,str.
        :return: The deciphered plaintext,str.
        >>> d = DecryptionUtils('key')
        >>> d.vigenere_decipher('ifmmp')
        'ybocl'

        """
        plaintext = ""
        for i, char in enumerate(ciphertext):
            if char in string.ascii_letters:
                plaintext += chr((ord(char) - ord('A') + ord(self.key[i % len(self.key)])) % 26 + ord('A'))
            else:
                plaintext += char
        return plaintext

    def rail_fence_decipher(self, encrypted_text, rails):
        """
        Deciphers the given ciphertext using the Rail Fence cipher
        :param encrypted_text: The ciphertext to decipher,str.
        :param rails: The number of rails to use for decryption,int.
        :return: The deciphered plaintext,str.
        >>> d = DecryptionUtils('key')
        >>> d.rail_fence_decipher('Hoo!el,Wrdl l', 3)
        'Hello, World!'

        """
        plaintext = ""
        for i in range(rails):
            for char in encrypted_text:
                if char in string.ascii_letters:
                    plaintext += chr((ord(char) - ord('A') + i) % 26 + ord('A'))
                else:
                    plaintext += char
        return plaintext

if __name__ == "__main__":
    # Test cases
    d = DecryptionUtils('key')
    output = d.caesar_decipher('ifmmp', 1)
    print(output)

    output = d.vigenere_decipher('ifmmp')
    print(output)

    output = d.rail_fence_decipher('Hoo!el,Wrdl l', 3)
    print(output)