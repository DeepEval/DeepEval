class BinaryDataProcessor:
    """
    This is a class used to process binary data, which includes functions such as clearing non 0 or 1 characters, counting binary string information, and converting to corresponding strings based on different encoding methods.
    """

    def __init__(self, binary_string):
        """
        Initialize the class with a binary string and clean it by removing all non 0 or 1 characters.
        """
        self.binary_string = binary_string
        self.clean_non_binary_chars()

    def clean_non_binary_chars(self):
        """
        Clean the binary string by removing all non 0 or 1 characters.
        >>> bdp = BinaryDataProcessor("01101000daf3e4r01100101011011000110110001101111")
        >>> bdp.clean_non_binary_chars()
        >>> bdp.binary_string
        '0110100001100101011011000110110001101111'

        """
        self.binary_string = ''.join([char for char in self.binary_string if char in '01'])

    def calculate_binary_info(self):
        """
        Calculate the binary string information, including the percentage of 0 and 1, and the total length of the binary string.
        >>> bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
        >>> bdp.calculate_binary_info()
        {'Zeroes': 0.475, 'Ones': 0.525, 'Bit length': 40}

        """
        zeroes = self.binary_string.count('0')
        ones = self.binary_string.count('1')
        bit_length = len(self.binary_string)
        return {'Zeroes': zeroes / bit_length, 'Ones': ones / bit_length, 'Bit length': bit_length}

    def convert_to_ascii(self):
        """
        Convert the binary string to ascii string.
        >>> bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
        >>> bdp.convert_to_ascii()
        'hello'

        """
        return ''.join([chr(int(self.binary_string[i:i + 8], 2)) for i in range(0, len(self.binary_string), 8)])

    def convert_to_utf8(self):
        """
        Convert the binary string to utf-8 string.
        >>> bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
        >>> bdp.convert_to_utf8()
        'hello'

        """
        utf8_list = []
        for i in range(0, len(self.binary_string), 8):
            byte = int(self.binary_string[i:i + 8], 2)
            if byte <= 0b0111111:
                utf8_list.append(chr(byte))
            elif byte <= 0b1111111 and byte >= 0b1100000:
                utf8_list.append(chr((byte >> 6) + 0b1100000))
                utf8_list.append(chr((byte & 0b11111) + 0b1000000))
            elif byte <= 0b11111111 and byte >= 0b11100000:
                utf8_list.append(chr((byte >> 12) + 0b1110000))
                utf8_list.append(chr(((byte >> 6) & 0b11111) + 0b1000000))
                utf8_list.append(chr((byte & 0b11111) + 0b1000000))
            else:
                raise ValueError("Invalid binary string")
        return ''.join(utf8_list)

if __name__ == "__main__":
    bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
    print(bdp.clean_non_binary_chars())
    print(bdp.calculate_binary_info())
    print(bdp.convert_to_ascii())
    print(bdp.convert_to_utf8())