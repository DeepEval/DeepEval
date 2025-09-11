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
                shift_amount = shift % 26
                if char.islower():
                    plaintext += chr((ord(char) - ord('a') - shift_amount) % 26 + ord('a'))
                else:
                    plaintext += chr((ord(char) - ord('A') - shift_amount) % 26 + ord('A'))
            else:
                plaintext += char
        return plaintext

    def vigenere_decipher(self, ciphertext):
        """
        Deciphers the given ciphertext using the Vigenere cipher
        :param ciphertext: The ciphertext to decipher, str.
        :return: The deciphered plaintext, str.
        """
        plaintext = ""
        key_length = len(self.key)
        key_as_num = [ord(char.upper()) - ord('A') for char in self.key]

        for i, char in enumerate(ciphertext):
            if char.isalpha():
                shift = key_as_num[i % key_length]
                if char.islower():
                    plaintext += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
                else:
                    plaintext += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
            else:
                plaintext += char
        return plaintext

    def rail_fence_decipher(self, encrypted_text, rails):
        """
        Deciphers the given ciphertext using the Rail Fence cipher
        :param encrypted_text: The ciphertext to decipher, str.
        :param rails: The number of rails to use for decryption, int.
        :return: The deciphered plaintext, str.
        """
        if rails <= 1 or rails >= len(encrypted_text):
            return encrypted_text

        # Create a list of strings for each rail
        rail_list = [''] * rails
        rail = 0
        direction = 1

        for char in encrypted_text:
            rail_list[rail] += char
            rail += direction
            if rail == 0 or rail == rails - 1:
                direction *= -1

        # Read the rail pattern from top to bottom
        plaintext = ''
        for i in range(rails):
            plaintext += rail_list[i]

        return plaintext


# Test cases
if __name__ == "__main__":
    d = DecryptionUtils('key')
    
    # Caesar Cipher Test
    print(d.caesar_decipher('ifmmp', 1))  # Expected output: 'hello'
    
    # Vigenere Cipher Test
    print(d.vigenere_decipher('ifmmp'))  # Expected output: 'ybocl'
    
    # Rail Fence Cipher Test
    print(d.rail_fence_decipher('Hoo!el,Wrdl l', 3))  # Expected output: 'Hello, World!'