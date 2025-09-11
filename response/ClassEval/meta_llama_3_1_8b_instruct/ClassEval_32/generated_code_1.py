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
            if char in string.ascii_lowercase:
                plaintext += chr((ord(char) - 97 - shift) % 26 + 97)
            elif char in string.ascii_uppercase:
                plaintext += chr((ord(char) - 65 - shift) % 26 + 65)
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
        key = self.key
        plaintext = ""
        key_index = 0
        for char in ciphertext:
            if char in string.ascii_lowercase:
                shift = ord(key[key_index % len(key)].lower()) - 97
                plaintext += chr((ord(char) - 97 - shift) % 26 + 97)
                key_index += 1
            elif char in string.ascii_uppercase:
                shift = ord(key[key_index % len(key)].lower()) - 97
                plaintext += chr((ord(char) - 65 - shift) % 26 + 65)
                key_index += 1
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
        # Find the middle rail
        middle_rail = (rails - 1) // 2

        # Create a grid with the given number of rails
        grid = [[''] * len(encrypted_text) for _ in range(rails)]

        # Distribute the characters in the encrypted text to the grid
        index = 0
        for i in range(rails):
            if i == middle_rail:
                continue
            step = 1 if i < middle_rail else -1
            for j in range(i, len(encrypted_text), rails + step):
                grid[i][j] = encrypted_text[index]
                index += 1

        # Read the characters from the grid to get the decrypted text
        decrypted_text = ''
        for col in range(len(encrypted_text)):
            for row in range(rails):
                if grid[row][col]!= '':
                    decrypted_text += grid[row][col]

        return decrypted_text

if __name__ == "__main__":
    d = DecryptionUtils('key')
    # Test case for caesar_decipher
    output = d.caesar_decipher('ifmmp', 1)
    print("Caesar Decipher:", output)

    # Test case for vigenere_decipher
    output = d.vigenere_decipher('ifmmp')
    print("Vigenere Decipher:", output)

    # Test case for rail_fence_decipher
    output = d.rail_fence_decipher('Hoo!el,Wrdl l', 3)
    print("Rail Fence Decipher:", output)