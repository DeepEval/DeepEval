import re

class NumericEntityUnescaper:
    def __init__(self):
        pass

    def replace(self, string):
        """
        Replaces numeric character references (HTML entities) in the input string
        with their corresponding Unicode characters.
        :param string: str, the input string containing numeric character references.
        :return: str, the input string with numeric character references replaced
        with their corresponding Unicode characters.
        """
        return re.sub(r'&#(\d+);', self.unescape_entity, string)

    @staticmethod
    def unescape_entity(match):
        """
        Unescapes a single numeric character reference.
        :param match: the match object for the numeric character reference.
        :return: the unescaped character.
        """
        code_point = int(match.group(1))
        return chr(code_point)

    @staticmethod
    def is_hex_char(char):
        """
        Determines whether a given character is a hexadecimal digit.
        :param char: str, the character to check.
        :return: bool, True if the character is a hexadecimal digit, False otherwise.
        """
        return '0' <= char <= '9' or 'a' <= char <= 'f' or 'A' <= char <= 'F'

if __name__ == "__main__":
    unescaper = NumericEntityUnescaper()
    string = "&#65;&#66;&#67;"
    output = unescaper.replace(string)
    print(output)