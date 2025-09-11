import re

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
        """
        def replace_match(match):
            entity = match.group(0)
            if entity.startswith('&#'):
                # Decimal numeric entity
                num = int(entity[2:-1])
            elif entity.startswith('&#x'):
                # Hexadecimal numeric entity
                num = int(entity[3:-1], 16)
            else:
                raise ValueError(f"Invalid numeric entity: {entity}")
            return chr(num)

        # Use regex to find all numeric entities
        pattern = re.compile(r'&#\d+;|&#x[0-9a-fA-F]+;')
        return pattern.sub(replace_match, string)

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
    output = unescaper.replace("&#65;&#66;&#67;")
    print(output)  # Should print 'ABC'

    # Test cases
    print(unescaper.replace("&#65;&#66;&#67;"))  # Should print 'ABC'
    print(unescaper.replace("&#104;&#101;&#108;&#108;&#111;"))  # Should print 'hello'
    print(unescaper.replace("&#x61;&#x62;&#x63;"))  # Should print 'abc'
    print(unescaper.replace("This is a test &#123; &#x34;."))  # Should print 'This is a test { 4.'

    # Test is_hex_char
    print(NumericEntityUnescaper.is_hex_char('a'))  # Should print True
    print(NumericEntityUnescaper.is_hex_char('g'))  # Should print False
    print(NumericEntityUnescaper.is_hex_char('0'))  # Should print True
    print(NumericEntityUnescaper.is_hex_char('9'))  # Should print True
    print(NumericEntityUnescaper.is_hex_char('F'))  # Should print True
    print(NumericEntityUnescaper.is_hex_char('G'))  # Should print False