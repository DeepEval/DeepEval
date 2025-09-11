import re
import binascii

class BinaryDataProcessor:
    def __init__(self, binary_string):
        self.binary_string = binary_string
        self.clean_non_binary_chars()

    def clean_non_binary_chars(self):
        self.binary_string = re.sub(r'[^01]', '', self.binary_string)

    def calculate_binary_info(self):
        zeroes = self.binary_string.count('0') / len(self.binary_string)
        ones = self.binary_string.count('1') / len(self.binary_string)
        bit_length = len(self.binary_string)
        return {'Zeroes': zeroes, 'Ones': ones, 'Bit length': bit_length}

    def convert_to_ascii(self):
        return binascii.unhexlify(self.binary_string).decode('utf-8')

    def convert_to_utf8(self):
        return self.binary_string.decode('utf-8')

# Test cases
if __name__ == "__main__":
    bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
    print(bdp.binary_string)  # Should print: 0110100001100101011011000110110001101111
    print(bdp.calculate_binary_info())  # Should print: {'Zeroes': 0.7, 'Ones': 0.3, 'Bit length': 40}
    print(bdp.convert_to_ascii())  # Should print: hello
    print(bdp.convert_to_utf8())  # Should print: hello