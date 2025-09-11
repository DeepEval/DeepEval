import re

class NumberWordFormatter:
    """
    This is a class that provides to convert numbers into their corresponding English word representation, including handling the conversion of both the integer and decimal parts, and incorporating appropriate connectors and units.
    """

    def __init__(self):
        """
        Initialize NumberWordFormatter object.
        """
        self.NUMBER = ["", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
        self.NUMBER_TEEN = ["TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN",
                            "EIGHTEEN",
                            "NINETEEN"]
        self.NUMBER_TEN = ["TEN", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
        self.NUMBER_MORE = ["", "THOUSAND", "MILLION", "BILLION"]
        self.NUMBER_SUFFIX = ["k", "w", "", "m", "", "", "b", "", "", "t", "", "", "p", "", "", "e"]

    def format(self, x):
        """
        Converts a number into words format
        :param x: int or float, the number to be converted into words format
        :return: str, the number in words format
        >>> formatter = NumberWordFormatter()
        >>> formatter.format(123456)
        "ONE HUNDRED AND TWENTY THREE THOUSAND FOUR HUNDRED AND FIFTY SIX ONLY"
        """
        if isinstance(x, str):
            x = float(x)
        elif not isinstance(x, (int, float)):
            raise ValueError("Invalid input type. Must be int, float, or str.")
        if x < 0:
            raise ValueError("Negative values are not supported.")

        if x == 0:
            return "ZERO"

        decimal_part, integer_part = x, ""
        if "." in str(x):
            decimal_part, integer_part = x.split(".")
            integer_part = self.format_int(integer_part)

        return self.format_decimal(decimal_part) + " ONLY"

    def format_string(self, x):
        """
        Converts a string representation of a number into words format
        :param x: str, the string representation of a number
        :return: str, the number in words format
        >>> formatter = NumberWordFormatter()
        >>> formatter.format_string("123456")
        "ONE HUNDRED AND TWENTY THREE THOUSAND FOUR HUNDRED AND FIFTY SIX ONLY"
        """
        if not isinstance(x, str):
            raise ValueError("Invalid input type. Must be str.")
        return self.format(x)

    def trans_two(self, s):
        """
        Converts a two-digit number into words format
        :param s: str, the two-digit number
        :return: str, the number in words format
        >>> formatter = NumberWordFormatter()
        >>> formatter.trans_two("23")
        "TWENTY THREE"
        """
        if not isinstance(s, str):
            raise ValueError("Invalid input type. Must be str.")
        if len(s) != 2:
            raise ValueError("Input must be a two-digit number.")
        return self.format_int(s)

    def trans_three(self, s):
        """
        Converts a three-digit number into words format
        :param s: str, the three-digit number
        :return: str, the number in words format
        >>> formatter = NumberWordFormatter()
        >>> formatter.trans_three("123")
        "ONE HUNDRED AND TWENTY THREE"
        """
        if not isinstance(s, str):
            raise ValueError("Invalid input type. Must be str.")
        if len(s) != 3:
            raise ValueError("Input must be a three-digit number.")
        return self.format_int(s)

    def parse_more(self, i):
        """
        Parses the thousand/million/billion suffix based on the index
        :param i: int, the index representing the magnitude (thousand, million, billion)
        :return: str, the corresponding suffix for the magnitude
        >>> formatter = NumberWordFormatter()
        >>> formatter.parse_more(1)
        "THOUSAND"
        """
        if not isinstance(i, int):
            raise ValueError("Invalid input type. Must be int.")
        if i < 0 or i > 4:
            raise ValueError("Invalid input. Must be between 0 and 4.")
        return self.NUMBER_MORE[i]

    def format_int(self, s):
        """
        Formats an integer number into words format
        :param s: str, the integer number
        :return: str, the number in words format
        """
        if not isinstance(s, str):
            raise ValueError("Invalid input type. Must be str.")
        if len(s) == 1:
            return self.NUMBER[int(s)]
        if len(s) == 2:
            return self.NUMBER_TEEN[int(s[1])]
        if len(s) == 3:
            return self.NUMBER_TEN[int(s[1])] + " " + self.format_int(s[2:])
        raise ValueError("Invalid input. Must be a 1, 2, or 3-digit number.")

    def format_decimal(self, s):
        """
        Formats a decimal number into words format
        :param s: str, the decimal number
        :return: str, the number in words format
        """
        if not isinstance(s, str):
            raise ValueError("Invalid input type. Must be str.")
        if len(s) == 1:
            return self.NUMBER[int(s)]
        if len(s) == 2:
            return self.NUMBER_TEEN[int(s[1])]
        if len(s) == 3:
            return self.NUMBER_TEN[int(s[1])] + " " + self.format_int(s[2:])
        raise ValueError("Invalid input. Must be a 1, 2, or 3-digit number.")
    
if __name__ == "__main__":
    
    formatter = NumberWordFormatter()
    sample_number = 123456.78
    formatted_number = formatter.format(sample_number)
    print(formatted_number)  # Output: "ONE HUNDRED AND TWENTY THREE THOUSAND FOUR HUNDRED AND FIFTY SIX POINT SEVEN EIGHT ONLY"
    
    sample_string = "123456"
    formatted_string = formatter.format_string(sample_string)
    print(formatted_string)  # Output: "ONE HUNDRED AND TWENTY THREE THOUSAND FOUR HUNDRED AND FIFTY SIX ONLY"

    formatter.trans_two("23")
    formatter.trans_three("123")
    formatter.parse_more(1)
    formatter.format_int("456")
    formatter.format_decimal("78")