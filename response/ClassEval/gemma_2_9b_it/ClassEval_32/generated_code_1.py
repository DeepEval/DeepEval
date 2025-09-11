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
        plaintext = ''
        for char in ciphertext:
            if char.isalpha():
                start = ord('a') if char.islower() else ord('A')
                shifted_char = chr((ord(char) - start - shift) % 26 + start)
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
        plaintext = ''
        key_len = len(self.key)
        key_as_int = [ord(i) for i in self.key]
        ciphertext_as_int = [ord(i) for i in ciphertext]
        for i in range(len(ciphertext_as_int)):
            value = (ciphertext_as_int[i] - key_as_int[i % key_len]) % 26
            plaintext += chr(value + ord('a'))
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
        plaintext = [''] * len(encrypted_text)
        rail = 0
        direction = 1
        for i, char in enumerate(encrypted_text):
            plaintext[rail] += char
            rail += direction
            if rail == rails - 1 or rail == 0:
                direction *= -1
        return ''.join(plaintext)

if __name__ == "__main__":
    instance = DecryptionUtils('key')
    # Test case for caesar_decipher
    ciphertext = 'ifmmp'
    shift = 1
    output = instance.caesar_decipher(ciphertext, shift)
    print(f"Caesar Decipher Output: {output}")

    # Test case for vigenere_decipher
    ciphertext = 'ifmmp'
    output = instance.vigenere_decipher(ciphertext)
    print(f"Vigenere Decipher Output: {output}")

    # Test case for rail_fence_decipher
    encrypted_text = 'Hoo!el,Wrdl l'
    rails = 3
    output = instance.rail_fence_decipher(encrypted_text, rails)
    print(f"Rail Fence Decipher Output: {output}")