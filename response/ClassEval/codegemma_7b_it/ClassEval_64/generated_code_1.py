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
        >>> NumberConverter.decimal_to_binary(42423)
        '1010010110110111'
        """
        if not isinstance(decimal_num, int):
            raise TypeError("Input number must be an integer")
        
        if decimal_num < 0:
            raise ValueError("Input number must be a non-negative integer")
        
        binary_num = ""
        while decimal_num > 0:
            remainder = decimal_num % 2
            binary_num = str(remainder) + binary_num
            decimal_num //= 2
        
        return binary_num

    @staticmethod
    def binary_to_decimal(binary_num):
        """
        Convert a number from binary format to decimal format.
        :param binary_num: str, binary number
        :return: int, the decimal representation of binary number str.
        >>> NumberConverter.binary_to_decimal('1010010110110111')
        42423
        """
        if not isinstance(binary_num, str):
            raise TypeError("Input number must be a string")
        
        if not all(bit in "01" for bit in binary_num):
            raise ValueError("Input number must contain only '0' and '1'")
        
        decimal_num = 0
        for i, bit in enumerate(binary_num):
            if bit == "1":
                decimal_num += 2 ** i
        
        return decimal_num

    @staticmethod
    def decimal_to_octal(decimal_num):
        """
        Convert a number from decimal format to octal format.
        :param decimal_num: int, decimal number
        :return: str, the octal representation of an integer.
        >>> NumberConverter.decimal_to_octal(42423)
        '122667'
        """
        if not isinstance(decimal_num, int):
            raise TypeError("Input number must be an integer")
        
        if decimal_num < 0:
            raise ValueError("Input number must be a non-negative integer")
        
        octal_num = ""
        while decimal_num > 0:
            remainder = decimal_num % 8
            octal_num = str(remainder) + octal_num
            decimal_num //= 8
        
        return octal_num

    @staticmethod
    def octal_to_decimal(octal_num):
        """
        Convert a number from octal format to decimal format.
        :param octal_num: str, octal num
        :return: int, the decimal representation of octal number str.
        >>> NumberConverter.octal_to_decimal('122667')
        42423
        """
        if not isinstance(octal_num, str):
            raise TypeError("Input number must be a string")
        
        if not all(bit in "01234567" for bit in octal_num):
            raise ValueError("Input number must contain only '0' to '7'")
        
        decimal_num = 0
        for i, bit in enumerate(octal_num):
            if bit == "1":
                decimal_num += 8 ** i
            elif bit == "2":
                decimal_num += 8 ** i * 2
            elif bit == "3":
                decimal_num += 8 ** i * 3
            elif bit == "4":
                decimal_num += 8 ** i * 4
            elif bit == "5":
                decimal_num += 8 ** i * 5
            elif bit == "6":
                decimal_num += 8 ** i * 6
            elif bit == "7":
                decimal_num += 8 ** i * 7
        
        return decimal_num

    @staticmethod
    def decimal_to_hex(decimal_num):
        """
        Convert a number from decimal format to hex format.
        :param decimal_num: int, decimal number
        :return hex_num: str, the hex representation of an integer.
        >>> NumberConverter.decimal_to_hex(42423)
        'a5b7'
        """
        if not isinstance(decimal_num, int):
            raise TypeError("Input number must be an integer")
        
        if decimal_num < 0:
            raise ValueError("Input number must be a non-negative integer")
        
        hex_num = ""
        while decimal_num > 0:
            remainder = decimal_num % 16
            if remainder < 10:
                hex_num = str(remainder) + hex_num
            else:
                hex_num = chr(ord('A') + remainder - 10) + hex_num
            decimal_num //= 16
        
        return hex_num

    @staticmethod
    def hex_to_decimal(hex_num):
        """
        Convert a number from hex format to decimal format.
        :param hex_num: str, hex num
        :return: int, the decimal representation of hex number str.
        >>> NumberConverter.hex_to_decimal('a5b7')
        42423
        """
        if not isinstance(hex_num, str):
            raise TypeError("Input number must be a string")
        
        if not all(bit in "0123456789abcdefABCDEF" for bit in hex_num):
            raise ValueError("Input number must contain only '0' to '9', 'a' to 'f', or 'A' to 'F'")
        
        decimal_num = 0
        for i, bit in enumerate(hex_num):
            if bit in "0123456789":
                decimal_num += int(bit) * 16 ** i
            elif bit in "abcdef":
                decimal_num += (ord(bit) - ord('a') + 10) * 16 ** i
            elif bit in "ABCDEF":
                decimal_num += (ord(bit) - ord('A') + 10) * 16 ** i
        
        return decimal_num

if __name__ == "__main__":
    # Test case for decimal_to_binary
    output = NumberConverter.decimal_to_binary(42423)
    print(output)

    # Test case for binary_to_decimal
    output = NumberConverter.binary_to_decimal("1010010110110111")
    print(output)

    # Test case for decimal_to_octal
    output = NumberConverter.decimal_to_octal(42423)
    print(output)

    # Test case for octal_to_decimal
    output = NumberConverter.octal_to_decimal("122667")
    print(output)

    # Test case for decimal_to_hex
    output = NumberConverter.decimal_to_hex(42423)
    print(output)

    # Test case for hex_to_decimal
    output = NumberConverter.hex_to_decimal("a5b7")
    print(output)