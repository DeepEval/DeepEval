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
        result = []
        i = 0
        while i < len(string):
            if string[i] == '#':
                start = i + 1
                end = string.find(';', start)
                if end != -1:
                    try:
                        codepoint = int(string[start:end])
                        result.append(chr(codepoint))
                        i = end + 1
                    except ValueError:
                        result.append(string[i])
                        i += 1
                else:
                    result.append(string[i])
                    i += 1
            else:
                result.append(string[i])
                i += 1
        return ''.join(result)

    @staticmethod
    def is_hex_char(char):
        """
        Determines whether a given character is a hexadecimal digit.
        :param char: str, the character to check.
        :return: bool, True if the character is a hexadecimal digit, False otherwise.
        >>> NumericEntityUnescaper.is_hex_char('a')
        True

        """
        return '0' <= char <= '9' or 'a' <= char <= 'f' or 'A' <= char <= 'F'



if __name__ == "__main__":
    unescaper = NumericEntityUnescaper()
    print(unescaper.replace("&#65;&#66;&#67;"))  # Output: ABC
    print(NumericEntityUnescaper.is_hex_char('a'))  # Output: True
    print(NumericEntityUnescaper.is_hex_char('G'))  # Output: False