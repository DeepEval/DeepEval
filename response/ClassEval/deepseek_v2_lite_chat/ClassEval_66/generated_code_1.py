import re

class NumericEntityUnescaper:
    """
    This is a class that provides functionality to replace numeric character references (HTML entities) with their corresponding Unicode characters.
    """

    @staticmethod
    def is_hex_char(char):
        """
        Determines whether a given character is a hexadecimal digit.
        :param char: str, the character to check.
        :return: bool, True if the character is a hexadecimal digit, False otherwise.
        """
        return re.match(r'[0-9a-fA-F]', char) is not None

    def replace(self, string):
        """
        Replaces numeric character references (HTML entities) in the input string with their corresponding Unicode characters.
        :param string: str, the input string containing numeric character references.
        :return: str, the input string with numeric character references replaced with their corresponding Unicode characters.
        """
        # Use regex to find all numeric character references
        entities = re.findall(r'&(#[\da-fA-F]+);', string)
        # Replace each entity with its corresponding Unicode character
        for entity in entities:
            code = int(entity[1:], 16)
            if not NumericEntityUnescaper.is_hex_char(chr(code)):
                raise ValueError(f"Invalid hexadecimal character reference: {chr(code)}")
            string = string.replace(entity, chr(code))
        return string

# Test cases
if __name__ == "__main__":
    unescaper = NumericEntityUnescaper()
    # Test case
    print(unescaper.replace("&#65;&#66;&#67;"))  # Should print 'ABC'
    # Test case with a non-hex character
    print(unescaper.replace("&#100000;"))  # Should raise ValueError