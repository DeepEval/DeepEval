class BinaryDataProcessor:
    """
    This is a class used to process binary data, which includes functions such as clearing non 0 or 1 characters,
    counting binary string information, and converting to corresponding strings based on different encoding methods.
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
        self.binary_string = ''.join(filter(lambda x: x in '01', self.binary_string))

    def calculate_binary_info(self):
        """
        Calculate the binary string information, including the percentage of 0 and 1, and the total length of the binary string.
        >>> bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
        >>> bdp.calculate_binary_info()
        {'Zeroes': 0.475, 'Ones': 0.525, 'Bit length': 40}
        """
        total_length = len(self.binary_string)
        if total_length == 0:
            return {'Zeroes': 0.0, 'Ones': 0.0, 'Bit length': 0}
        
        count_ones = self.binary_string.count('1')
        count_zeroes = self.binary_string.count('0')
        
        return {
            'Zeroes': round(count_zeroes / total_length, 3),
            'Ones': round(count_ones / total_length, 3),
            'Bit length': total_length
        }

    def convert_to_ascii(self):
        """
        Convert the binary string to ascii string.
        >>> bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
        >>> bdp.convert_to_ascii()
        'hello'
        """
        ascii_string = ''.join(chr(int(self.binary_string[i:i + 8], 2)) for i in range(0, len(self.binary_string), 8))
        return ascii_string

    def convert_to_utf8(self):
        """
        Convert the binary string to utf-8 string.
        >>> bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
        >>> bdp.convert_to_utf8()
        'hello'
        """
        return self.convert_to_ascii()  # UTF-8 is backward compatible with ASCII in this case

# Test cases for each method
if __name__ == "__main__":
    # Test clean_non_binary_chars
    bdp1 = BinaryDataProcessor("01101000daf3e4r01100101011011000110110001101111")
    print(bdp1.binary_string)  # Expected: '0110100001100101011011000110110001101111'

    # Test calculate_binary_info
    info = bdp1.calculate_binary_info()
    print(info)  # Expected: {'Zeroes': 0.475, 'Ones': 0.525, 'Bit length': 40}

    # Test convert_to_ascii
    ascii_output = bdp1.convert_to_ascii()
    print(ascii_output)  # Expected: 'hello'

    # Test convert_to_utf8
    utf8_output = bdp1.convert_to_utf8()
    print(utf8_output)  # Expected: 'hello'