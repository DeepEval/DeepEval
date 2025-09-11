class DecryptionUtils:
    """
    This is a class that provides methods for decryption, including the Caesar cipher, Vigenere cipher, and Rail Fence cipher.
    """

    def __init__(self, key):
        """
        Initializes the decryption utility with a key.
        :param key: The key to use for decryption, str.
        """
        self.key = key

    def caesar_decipher(self, ciphertext, shift):
        """
        Deciphers the given ciphertext using the Caesar cipher
        :param ciphertext: The ciphertext to decipher, str.
        :param shift: The shift to use for decryption, int.
        :return: The deciphered plaintext, str.
        """
        plaintext = ""
        for char in ciphertext:
            if char.isalpha():
                shifted = ord(char) - shift
                if char.islower():
                    if shifted < ord('a'):
                        shifted += 26
                elif char.isupper():
                    if shifted < ord('A'):
                        shifted += 26
                plaintext += chr(shifted)
            else:
                plaintext += char
        return plaintext

    def vigenere_decipher(self, ciphertext):
        """
        Deciphers the given ciphertext using the Vigenere cipher
        :param ciphertext: The ciphertext to decipher, str.
        :return: The deciphered plaintext, str.
        """
        plaintext = []
        key_length = len(self.key)
        key_index = 0

        for char in ciphertext:
            if char.isalpha():
                shift = ord(self.key[key_index % key_length].lower()) - ord('a')
                if char.islower():
                    deciphered_char = chr((ord(char) - shift - ord('a')) % 26 + ord('a'))
                else:
                    deciphered_char = chr((ord(char) - shift - ord('A')) % 26 + ord('A'))
                plaintext.append(deciphered_char)
                key_index += 1
            else:
                plaintext.append(char)

        return ''.join(plaintext)

    def rail_fence_decipher(self, encrypted_text, rails):
        """
        Deciphers the given ciphertext using the Rail Fence cipher
        :param encrypted_text: The ciphertext to decipher, str.
        :param rails: The number of rails to use for decryption, int.
        :return: The deciphered plaintext, str.
        """
        n = len(encrypted_text)
        rail = [['\n' for _ in range(n)] for _ in range(rails)]
        direction_down = False
        row, col = 0, 0

        for char in encrypted_text:
            if row == 0:
                direction_down = True
            if row == rails - 1:
                direction_down = False

            rail[row][col] = '*'
            col += 1

            if direction_down:
                row += 1
            else:
                row -= 1

        index = 0
        for i in range(rails):
            for j in range(n):
                if (rail[i][j] == '*' and index < len(encrypted_text)):
                    rail[i][j] = encrypted_text[index]
                    index += 1

        result = []
        row, col = 0, 0

        for char in encrypted_text:
            if row == 0:
                direction_down = True
            if row == rails - 1:
                direction_down = False

            if rail[row][col] != '*':
                result.append(rail[row][col])
                col += 1

            if direction_down:
                row += 1
            else:
                row -= 1

        return ''.join(result)

# Testing the methods
if __name__ == "__main__":
    d = DecryptionUtils('key')

    # Test case for Caesar cipher
    caesar_output = d.caesar_decipher('ifmmp', 1)
    print(f'Caesar Decipher Output: {caesar_output}')  # Expected: hello

    # Test case for Vigenere cipher
    vigenere_output = d.vigenere_decipher('ifmmp')
    print(f'Vigenere Decipher Output: {vigenere_output}')  # Expected: ybocl

    # Test case for Rail Fence cipher
    rail_fence_output = d.rail_fence_decipher('Hoo!el,Wrdl l', 3)
    print(f'Rail Fence Decipher Output: {rail_fence_output}')  # Expected: Hello, World!