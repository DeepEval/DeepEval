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
        """
        deciphered_text = []
        for char in ciphertext:
            if char.isalpha():
                shift_amount = shift % 26
                new_char = chr((ord(char) - shift_amount - 97) % 26 + 97) if char.islower() else chr((ord(char) - shift_amount - 65) % 26 + 65)
                deciphered_text.append(new_char)
            else:
                deciphered_text.append(char)
        return ''.join(deciphered_text)

    def vigenere_decipher(self, ciphertext):
        """
        Deciphers the given ciphertext using the Vigenere cipher
        :param ciphertext: The ciphertext to decipher,str.
        :return: The deciphered plaintext,str.
        """
        deciphered_text = []
        key_length = len(self.key)
        for i, char in enumerate(ciphertext):
            if char.isalpha():
                shift_amount = ord(self.key[i % key_length].lower()) - 97
                new_char = chr((ord(char) - shift_amount - 97) % 26 + 97) if char.islower() else chr((ord(char) - shift_amount - 65) % 26 + 65)
                deciphered_text.append(new_char)
            else:
                deciphered_text.append(char)
        return ''.join(deciphered_text)

    def rail_fence_decipher(self, encrypted_text, rails):
        """
        Deciphers the given ciphertext using the Rail Fence cipher
        :param encrypted_text: The ciphertext to decipher,str.
        :param rails: The number of rails to use for decryption,int.
        :return: The deciphered plaintext,str.
        """
        if rails <= 1:
            return encrypted_text

        # Create an empty matrix to store the zigzag pattern
        matrix = [['\n' for _ in range(len(encrypted_text))] for _ in range(rails)]
        dir_down = None
        row, col = 0, 0

        # Identify the positions with characters in the zigzag pattern
        for char in encrypted_text:
            if row == 0:
                dir_down = True
            if row == rails - 1:
                dir_down = False
            matrix[row][col] = '*'
            col += 1
            row += 1 if dir_down else -1

        # Fill the zigzag pattern with actual characters
        index = 0
        for i in range(rails):
            for j in range(len(encrypted_text)):
                if matrix[i][j] == '*' and index < len(encrypted_text):
                    matrix[i][j] = encrypted_text[index]
                    index += 1

        # Read the message by following the zigzag pattern
        result = []
        row, col = 0, 0
        for char in encrypted_text:
            if row == 0:
                dir_down = True
            if row == rails - 1:
                dir_down = False
            if matrix[row][col] != '*':
                result.append(matrix[row][col])
                col += 1
            row += 1 if dir_down else -1

        return ''.join(result)

if __name__ == "__main__":
    # Test cases
    d = DecryptionUtils('key')

    # Test Caesar Decipher
    output = d.caesar_decipher('ifmmp', 1)
    print("Caesar Decipher:", output)  # Expected: 'hello'

    # Test Vigenere Decipher
    output = d.vigenere_decipher('ifmmp')
    print("Vigenere Decipher:", output)  # Expected: 'ybocl'

    # Test Rail Fence Decipher
    output = d.rail_fence_decipher('Hoo!el,Wrdl l', 3)
    print("Rail Fence Decipher:", output)  # Expected: 'Hello, World!'