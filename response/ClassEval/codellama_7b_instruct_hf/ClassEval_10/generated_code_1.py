import re

class BinaryDataProcessor:
    def __init__(self, binary_string):
        self.binary_string = binary_string

    def clean_non_binary_chars(self):
        self.binary_string = re.sub(r'[^01]', '', self.binary_string)

    def calculate_binary_info(self):
        binary_len = len(self.binary_string)
        zero_count = self.binary_string.count('0')
        one_count = binary_len - zero_count
        return {
            'Zeroes': zero_count / binary_len,
            'Ones': one_count / binary_len,
            'Bit length': binary_len
        }

    def convert_to_ascii(self):
        return bytes(self.binary_string, 'ascii').decode('ascii')

    def convert_to_utf8(self):
        return bytes(self.binary_string, 'utf-8').decode('utf-8')

# Test cases
if __name__ == "__main__":
    bdp = BinaryDataProcessor("0110100001100101011011000110110001101111")
    bdp.clean_non_binary_chars()
    print(bdp.calculate_binary_info())
    print(bdp.convert_to_ascii())
    print(bdp.convert_to_utf8())