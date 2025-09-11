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
        self.binary_string = ''.join(char for char in self.binary_string if char in ['0', '1'])

    def calculate_binary_info(self):
        """
        Calculate the binary string information, including the percentage of 0 and 1, and the total length of the binary string.
        >>> bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
        >>> bdp.calculate_binary_info()
        {'Zeroes': 0.475, 'Ones': 0.525, 'Bit length': 40}

        """
        zero_count = self.binary_string.count('0')
        one_count = self.binary_string.count('1')
        total_length = len(self.binary_string)
        return {
            'Zeroes': zero_count / total_length,
            'Ones': one_count / total_length,
            'Bit length': total_length
        }

    def convert_to_ascii(self):
        """
        Convert the binary string to ascii string.
        >>> bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
        >>> bdp.convert_to_ascii()
        'hello'

        """
        # Convert binary string to bytes and then decode it to ascii string
        ascii_string = bytes(int(self.binary_string[i:i+8], 2) for i in range(0, len(self.binary_string), 8)).decode('ascii')
        return ascii_string

    def convert_to_utf8(self):
        """
        Convert the binary string to utf-8 string.
        >>> bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
        >>> bdp.convert_to_utf8()
        'hello'

        """
        # Convert binary string to bytes and then decode it to utf-8 string
        utf8_string = bytes(int(self.binary_string[i:i+8], 2) for i in range(0, len(self.binary_string), 8)).decode('utf-8')
        return utf8_string

if __name__ == "__main__":
    instance = BinaryDataProcessor("0110100001100101011011000110110001101111")
    
    # Test case for clean_non_binary_chars method
    print("Cleaned binary string:", instance.binary_string)
    
    # Test case for calculate_binary_info method
    binary_info = instance.calculate_binary_info()
    print("Binary string information:", binary_info)
    
    # Test case for convert_to_ascii method
    ascii_string = instance.convert_to_ascii()
    print("Ascii string:", ascii_string)
    
    # Test case for convert_to_utf8 method
    utf8_string = instance.convert_to_utf8()
    print("Utf-8 string:", utf8_string)