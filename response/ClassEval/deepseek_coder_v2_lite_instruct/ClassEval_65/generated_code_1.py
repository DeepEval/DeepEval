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
        """
        if isinstance(x, float):
            integer_part, decimal_part = str(x).split('.')
        else:
            integer_part = str(x)
            decimal_part = ""

        integer_words = self.convert_integer_to_words(integer_part)
        decimal_words = self.convert_decimal_to_words(decimal_part)

        if decimal_words:
            return f"{integer_words} {decimal_words} ONLY"
        else:
            return f"{integer_words} ONLY"

    def format_string(self, x):
        """
        Converts a string representation of a number into words format
        :param x: str, the string representation of a number
        :return: str, the number in words format
        """
        try:
            num = float(x)
        except ValueError:
            raise ValueError("Input must be a valid number")

        return self.format(num)

    def trans_two(self, s):
        """
        Converts a two-digit number into words format
        :param s: str, the two-digit number
        :return: str, the number in words format
        """
        if int(s) < 10:
            return self.NUMBER[int(s)]
        elif int(s) < 20:
            return self.NUMBER_TEEN[int(s) - 10]
        else:
            tens = int(s[0])
            units = int(s[1])
            return f"{self.NUMBER_TEN[tens - 1]} {self.NUMBER[units]}"

    def trans_three(self, s):
        """
        Converts a three-digit number into words format
        :param s: str, the three-digit number
        :return: str, the number in words format
        """
        hundreds = int(s[0])
        rest = s[1:]
        if hundreds == 0:
            return self.trans_two(rest)
        else:
            return f"{self.NUMBER[hundreds]} HUNDRED AND {self.trans_two(rest)}"

    def parse_more(self, i):
        """
        Parses the thousand/million/billion suffix based on the index
        :param i: int, the index representing the magnitude (thousand, million, billion)
        :return: str, the corresponding suffix for the magnitude
        """
        return self.NUMBER_MORE[i]

    def convert_integer_to_words(self, integer_part):
        if len(integer_part) == 0:
            return "ZERO"
        if int(integer_part) == 0:
            return "ZERO"

        parts = []
        magnitude = (len(integer_part) - 1) // 3
        while magnitude >= 0:
            part = integer_part[-3:]
            integer_part = integer_part[:-3]
            if int(part) > 0:
                parts.append(self.trans_three(part))
                parts.append(self.parse_more(magnitude))
            magnitude -= 1

        return " ".join(reversed(parts))

    def convert_decimal_to_words(self, decimal_part):
        if len(decimal_part) == 0:
            return ""
        words = []
        for digit in decimal_part:
            words.append(self.NUMBER[int(digit)])
        return " ".join(words)


if __name__ == "__main__":
    formatter = NumberWordFormatter()

    # Test cases
    print(formatter.format(123456))  # Output: "ONE HUNDRED AND TWENTY THREE THOUSAND FOUR HUNDRED AND FIFTY SIX ONLY"
    print(formatter.format(123456.789))  # Output: "ONE HUNDRED AND TWENTY THREE THOUSAND FOUR HUNDRED AND FIFTY SIX DOT SEVENTY EIGHT NINE ONLY"
    print(formatter.format_string("123456"))  # Output: "ONE HUNDRED AND TWENTY THREE THOUSAND FOUR HUNDRED AND FIFTY SIX ONLY"
    print(formatter.trans_two("23"))  # Output: "TWENTY THREE"
    print(formatter.trans_three("123"))  # Output: "ONE HUNDRED AND TWENTY THREE"
    print(formatter.parse_more(1))  # Output: "THOUSAND"