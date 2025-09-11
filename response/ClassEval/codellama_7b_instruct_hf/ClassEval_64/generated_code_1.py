import binascii

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
        return hex(decimal_num).replace("0x", "")

    @staticmethod
    def hex_to_decimal(hex_num):
        return int(hex_num, 16)

if __name__ == "__main__":
    instance = NumberConverter()
    print(instance.decimal_to_binary(42423))
    print(instance.binary_to_decimal("1010010110110111"))
    print(instance.decimal_to_octal(42423))
    print(instance.octal_to_decimal("122667"))
    print(instance.decimal_to_hex(42423))
    print(instance.hex_to_decimal("a5b7"))