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
        """
        self.binary_string = ''.join(filter(lambda x: x in '01', self.binary_string))

    def calculate_binary_info(self):
        """
        Calculate the binary string information, including the percentage of 0 and 1, and the total length of the binary string.
        """
        total_length = len(self.binary_string)
        zeroes = self.binary_string.count('0')
        ones = self.binary_string.count('1')
        percentage_zeroes = zeroes / total_length
        percentage_ones = ones / total_length
        return {
            'Zeroes': round(percentage_zeroes, 3),
            'Ones': round(percentage_ones, 3),
            'Bit length': total_length
        }

    def convert_to_ascii(self):
        """
        Convert the binary string to ascii string.
        """
        return ''.join(chr(int(self.binary_string[i:i+8], 2)) for i in range(0, len(self.binary_string), 8))

    def convert_to_utf8(self):
        """
        Convert the binary string to utf-8 string.
        """
        utf8_bytes = bytearray(int(self.binary_string[i:i+8], 2) for i in range(0, len(self.binary_string), 8))
        return utf8_bytes.decode('utf-8')

if __name__ == "__main__":
    # Test cases
    bdp = BinaryDataProcessor("01101000daf3e4r01100101011011000110110001101111")
    bdp.clean_non_binary_chars()
    print("Cleaned Binary String:", bdp.binary_string)  # Expected: '0110100001100101011011000110110001101111'

    bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
    info = bdp.calculate_binary_info()
    print("Binary Information:", info)  # Expected: {'Zeroes': 0.475, 'Ones': 0.525, 'Bit length': 40}

    bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
    ascii_string = bdp.convert_to_ascii()
    print("ASCII String:", ascii_string)  # Expected: 'hello'

    bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
    utf8_string = bdp.convert_to_utf8()
    print("UTF-8 String:", utf8_string)  # Expected: 'hello'