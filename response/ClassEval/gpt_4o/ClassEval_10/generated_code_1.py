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
        length = len(self.binary_string)
        if length == 0:
            return {'Zeroes': 0.0, 'Ones': 0.0, 'Bit length': 0}
        
        zero_count = self.binary_string.count('0')
        one_count = self.binary_string.count('1')
        zero_percentage = zero_count / length
        one_percentage = one_count / length

        return {'Zeroes': zero_percentage, 'Ones': one_percentage, 'Bit length': length}

    def convert_to_ascii(self):
        """
        Convert the binary string to ascii string.
        """
        ascii_string = ''
        for i in range(0, len(self.binary_string), 8):
            byte = self.binary_string[i:i+8]
            if len(byte) == 8:
                ascii_string += chr(int(byte, 2))
        return ascii_string

    def convert_to_utf8(self):
        """
        Convert the binary string to utf-8 string.
        """
        return self.convert_to_ascii()  # ASCII and UTF-8 are compatible for this range of characters.

if __name__ == "__main__":
    # Test case for clean_non_binary_chars
    bdp = BinaryDataProcessor("01101000daf3e4r01100101011011000110110001101111")
    print("Cleaned Binary String:", bdp.binary_string)  # Expected: '0110100001100101011011000110110001101111'

    # Test case for calculate_binary_info
    bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
    binary_info = bdp.calculate_binary_info()
    print("Binary Info:", binary_info)  # Expected: {'Zeroes': 0.475, 'Ones': 0.525, 'Bit length': 40}

    # Test case for convert_to_ascii
    bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
    ascii_string = bdp.convert_to_ascii()
    print("ASCII String:", ascii_string)  # Expected: 'hello'

    # Test case for convert_to_utf8
    utf8_string = bdp.convert_to_utf8()
    print("UTF-8 String:", utf8_string)  # Expected: 'hello'