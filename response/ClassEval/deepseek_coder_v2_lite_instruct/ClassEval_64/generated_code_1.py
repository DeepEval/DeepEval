class NumberConverter:
    """
    The class allows to convert  decimal to binary, octal and hexadecimal repectively and contrarily
    """

    @staticmethod
    def decimal_to_binary(decimal_num):
        """
        Convert a number from decimal format to binary format.
        :param decimal_num: int, decimal number
        :return: str, the binary representation of an integer.
        """
        return bin(decimal_num)[2:]

    @staticmethod
    def binary_to_decimal(binary_num):
        """
        Convert a number from binary format to decimal format.
        :param binary_num: str, binary number
        :return: int, the decimal representation of binary number str.
        """
        return int(binary_num, 2)

    @staticmethod
    def decimal_to_octal(decimal_num):
        """
        Convert a number from decimal format to octal format.
        :param decimal_num: int, decimal number
        :return: str, the octal representation of an integer.
        """
        return oct(decimal_num)[2:]

    @staticmethod
    def octal_to_decimal(octal_num):
        """
        Convert a number from octal format to decimal format.
        :param octal_num: str, octal num
        :return: int, the decimal representation of octal number str.
        """
        return int(octal_num, 8)

    @staticmethod
    def decimal_to_hex(decimal_num):
        """
        Convert a number from decimal format to hex format.
        :param decimal_num: int, decimal number
        :return: str, the hex representation of an integer.
        """
        return hex(decimal_num)[2:]

    @staticmethod
    def hex_to_decimal(hex_num):
        """
        Convert a number from hex format to decimal format.
        :param hex_num: str, hex num
        :return: int, the decimal representation of hex number str.
        """
        return int(hex_num, 16)


if __name__ == "__main__":
    # Test cases
    print("Binary to Decimal:", NumberConverter.binary_to_decimal('1010010110110111'))  # Output: 42423
    print("Decimal to Binary:", NumberConverter.decimal_to_binary(42423))  # Output: '1010010110110111'
    print("Decimal to Octal:", NumberConverter.decimal_to_octal(42423))  # Output: '122667'
    print("Octal to Decimal:", NumberConverter.octal_to_decimal('122667'))  # Output: 42423
    print("Decimal to Hex:", NumberConverter.decimal_to_hex(42423))  # Output: 'a5b7'
    print("Hex to Decimal:", NumberConverter.hex_to_decimal('a5b7'))  # Output: 42423