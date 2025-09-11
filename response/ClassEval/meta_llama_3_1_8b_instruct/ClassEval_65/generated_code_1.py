class NumberWordFormatter:
    """
    This is a class that provides to convert numbers into their corresponding English word representation, 
    including handling the conversion of both the integer and decimal parts, and incorporating appropriate 
    connectors and units.
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
        if isinstance(x, float):
            x = round(x, 2)
        if isinstance(x, int):
            return self.format_int(x)
        else:
            raise ValueError("Input must be an integer or a float.")

    def format_string(self, x):
        """
        Converts a string representation of a number into words format
        :param x: str, the string representation of a number
        :return: str, the number in words format
        >>> formatter = NumberWordFormatter()
        >>> formatter.format_string("123456")
        "ONE HUNDRED AND TWENTY THREE THOUSAND FOUR HUNDRED AND FIFTY SIX ONLY"
        """
        try:
            return self.format(int(x))
        except ValueError:
            raise ValueError("Invalid input string.")

    def trans_two(self, s):
        """
        Converts a two-digit number into words format
        :param s: str, the two-digit number
        :return: str, the number in words format
        >>> formatter = NumberWordFormatter()
        >>> formatter.trans_two("23")
        "TWENTY THREE"
        """
        try:
            num = int(s)
            if num < 10:
                return self.NUMBER[num]
            elif num < 20:
                return self.NUMBER_TEEN[num - 10]
            else:
                tens, ones = divmod(num, 10)
                return self.NUMBER_TEN[tens - 1] + " " + self.NUMBER[ones]
        except ValueError:
            raise ValueError("Invalid input string.")

    def trans_three(self, s):
        """
        Converts a three-digit number into words format
        :param s: str, the three-digit number
        :return: str, the number in words format
        >>> formatter = NumberWordFormatter()
        >>> formatter.trans_three("123")
        "ONE HUNDRED AND TWENTY THREE"
        """
        try:
            num = int(s)
            if num < 100:
                return self.trans_two(str(num))
            else:
                hundreds, rest = divmod(num, 100)
                return self.NUMBER[hundreds] + " HUNDRED " + self.trans_two(str(rest))
        except ValueError:
            raise ValueError("Invalid input string.")

    def parse_more(self, i):
        """
        Parses the thousand/million/billion suffix based on the index
        :param i: int, the index representing the magnitude (thousand, million, billion)
        :return: str, the corresponding suffix for the magnitude
        >>> formatter = NumberWordFormatter()
        >>> formatter.parse_more(1)
        "THOUSAND"
        """
        return self.NUMBER_MORE[i]

    def format_int(self, x):
        """
        Converts an integer into words format
        :param x: int, the integer to be converted into words format
        :return: str, the number in words format
        """
        if x == 0:
            return "ZERO"
        elif x < 0:
            return "MINUS " + self.format_int(-x)
        elif x < 100:
            return self.trans_two(str(x))
        elif x < 1000:
            hundreds, rest = divmod(x, 100)
            return self.trans_two(str(hundreds)) + " HUNDRED " + self.trans_two(str(rest))
        elif x < 1000000:
            thousands, rest = divmod(x, 1000)
            return self.format_int(thousands) + " " + self.parse_more(1) + " " + self.format_int(rest)
        elif x < 1000000000:
            millions, rest = divmod(x, 1000000)
            return self.format_int(millions) + " " + self.parse_more(2) + " " + self.format_int(rest)
        else:
            billions, rest = divmod(x, 1000000000)
            return self.format_int(billions) + " " + self.parse_more(3) + " " + self.format_int(rest)

if __name__ == "__main__":
    formatter = NumberWordFormatter()

    # Test case for format
    print(formatter.format(123456))
    print(formatter.format(123456.789))

    # Test case for format_string
    print(formatter.format_string("123456"))
    print(formatter.format_string("123456.789"))

    # Test case for trans_two
    print(formatter.trans_two("23"))
    print(formatter.trans_two("12"))
    print(formatter.trans_two("99"))

    # Test case for trans_three
    print(formatter.trans_three("123"))
    print(formatter.trans_three("012"))
    print(formatter.trans_three("999"))

    # Test case for parse_more
    print(formatter.parse_more(1))
    print(formatter.parse_more(2))
    print(formatter.parse_more(3))