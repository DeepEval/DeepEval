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
            if char.isalpha():
                if char.islower():
                    base = ord('a')
                else:
                    base = ord('A')
                shifted_char = chr((ord(char) - base - shift) % 26 + base)
            else:
                shifted_char = char
            plaintext += shifted_char
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
        key_length = len(self.key)
        plaintext = ""
        for i, char in enumerate(ciphertext):
            if char.isalpha():
                if char.islower():
                    base = ord('a')
                else:
                    base = ord('A')
                key_char = self.key[i % key_length]
                shift = ord(key_char) - base
                shifted_char = chr((ord(char) - base - shift) % 26 + base)
            else:
                shifted_char = char
            plaintext += shifted_char
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
        if rails <= 1:
            return encrypted_text

        rail_length = (len(encrypted_text) + rails - 2) // (rails - 1)
        rail_positions = [i for i in range(rails)]
        rail_positions.reverse()

        plaintext = [''] * len(encrypted_text)
        current_rail = 0
        going_down = True

        for i, char in enumerate(encrypted_text):
            plaintext[rail_positions[current_rail]] += char
            if current_rail == 0 or current_rail == rails - 1:
                going_down = not going_down
            if going_down:
                current_rail += 1
            else:
                current_rail -= 1

        return ''.join(plaintext)

if __name__ == "__main__":
    d = DecryptionUtils('key')
    print(d.caesar_decipher('ifmmp', 1))
    print(d.vigenere_decipher('ifmmp'))
    print(d.rail_fence_decipher('Hoo!el,Wrdl l', 3))