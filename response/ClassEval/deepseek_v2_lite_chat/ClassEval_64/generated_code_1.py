class NumberConverter:
    @staticmethod
    def decimal_to_binary(decimal_num):
        return bin(decimal_num).replace("0b", "")

    @staticmethod
    def binary_to_decimal(binary_num):
        return int(binary_num, 2)

    @staticmethod
    def decimal_to_octal(decimal_num):
        return oct(decimal_num).replace("0o", "")

    @staticmethod
    def octal_to_decimal(octal_num):
        return int(octal_num, 8)

    @staticmethod
    def decimal_to_hex(decimal_num):
        return hex(decimal_num).replace("0x", "").upper()

    @staticmethod
    def hex_to_decimal(hex_num):
        return int(hex_num, 16)


if __name__ == "__main__":
    # Test cases for decimal_to_binary
    print(NumberConverter.decimal_to_binary(42423))  # Output: '1010010110110111'
    print(NumberConverter.decimal_to_binary(2))       # Output: '10'

    # Test cases for binary_to_decimal
    print(NumberConverter.binary_to_decimal('1010010110110111'))  # Output: 42423
    print(NumberConverter.binary_to_decimal('1111'))             # Output: 15

    # Test cases for decimal_to_octal
    print(NumberConverter.decimal_to_octal(42423))  # Output: '122667'
    print(NumberConverter.decimal_to_octal(8))       # Output: '10'

    # Test cases for octal_to_decimal
    print(NumberConverter.octal_to_decimal('122667'))  # Output: 42423
    print(NumberConverter.octal_to_decimal('777'))     # Output: 4095

    # Test cases for decimal_to_hex
    print(NumberConverter.decimal_to_hex(42423))  # Output: 'a5b7'
    print(NumberConverter.decimal_to_hex(10))     # Output: 'a'

    # Test cases for hex_to_decimal
    print(NumberConverter.hex_to_decimal('a5b7'))  # Output: 42423
    print(NumberConverter.hex_to_decimal('f0f'))    # Output: 4095