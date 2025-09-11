class NumericEntityUnescaper:
    """
    This is a class that provides functionality to replace numeric entities with their corresponding characters in a given string.
    """

    def __init__(self):
        pass

    def replace(self, string):
        """
        Replaces numeric character references (HTML entities) in the input string with their corresponding Unicode characters.
        :param string: str, the input string containing numeric character references.
        :return: str, the input string with numeric character references replaced with their corresponding Unicode characters.
        >>> unescaper = NumericEntityUnescaper()
        >>> unescaper.replace("&#65;&#66;&#67;")
        'ABC'

        """
        output = ""
        for char in string:
            if char == ";":
                break
            if char == "#":
                hex_value = ""
                while True:
                    next_char = string[string.index(char) + 1]
                    if NumericEntityUnescaper.is_hex_char(next_char):
                        hex_value += next_char
                        string = string[string.index(next_char) + 1:]
                    else:
                        break
                output += chr(int(hex_value, 16))
            else:
                output += char
        return output

    @staticmethod
    def is_hex_char(char):
        """
        Determines whether a given character is a hexadecimal digit.
        :param char: str, the character to check.
        :return: bool, True if the character is a hexadecimal digit, False otherwise.
        >>> NumericEntityUnescaper.is_hex_char('a')
        True

        """
        return char in "0123456789abcdefABCDEF"


if __name__ == "__main__":
    unescaper = NumericEntityUnescaper()
    output = unescaper.replace("&#65;&#66;&#67;")
    print(output)