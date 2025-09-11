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
        """
        integer, decimal = int(x), float('%.2f' % (x % 1)) if x != int(x) else (0, 0)
        words = []

        if integer:
            words.append(self._format_integer(integer))

        if decimal > 0:
            words.append('AND')
            words.append(self._format_decimal(decimal))

        return ' '.join(words).strip()

    def format_string(self, x):
        """
        Converts a string representation of a number into words format
        :param x: str, the string representation of a number
        :return: str, the number in words format
        """
        return self.format(float(x))

    def trans_two(self, s):
        """
        Converts a two-digit number into words format
        :param s: str, the two-digit number
        :return: str, the number in words format
        """
        return ' '.join(self.NUMBER[int(s)]).strip()

    def trans_three(self, s):
        """
        Converts a three-digit number into words format
        :param s: str, the three-digit number
        :return: str, the number in words format
        """
        parts = s.split('.')
        integer, decimal = int(parts[0]), float('0.' + parts[1])
        return self.trans_two(str(integer)) + ' ' + self._format_decimal(decimal)

    def parse_more(self, i):
        """
        Parses the thousand/million/billion suffix based on the index
        :param i: int, the index representing the magnitude (thousand, million, billion)
        :return: str, the corresponding suffix for the magnitude
        """
        return self.NUMBER_MORE[i]

    def _format_integer(self, n):
        """
        Helper method to format the integer part of a number.
        :param n: int, the integer part of the number
        :return: str, the word representation of the integer
        """
        if n < 10:
            return self.NUMBER[n]
        elif 10 <= n < 20:
            return self.NUMBER_TEEN[n - 10]
        else:
            tens, units = divmod(n, 10)
            return self.NUMBER_TEN[tens - 1] + ('' if units == 0 else ' ' + self.NUMBER[units])

    def _format_decimal(self, d):
        """
        Helper method to format the decimal part of a number.
        :param d: float, the decimal part of the number
        :return: str, the word representation of the decimal
        """
        suffix = ''
        if d > 0:
            suffix = self.NUMBER_SUFFIX[len(self.NUMBER_SUFFIX) - 1]
            d *= 1000
            d, i = divmod(d, 1000)
            d, i = divmod(d, 1000)
            d, i = divmod(d, 1000)
        return suffix + ('%d%s' % (d, suffix) if i > 0 else '.%d%s' % (d, suffix))


if __name__ == "__main__":
    formatter = NumberWordFormatter()
    print(formatter.format(123456))  # Should print "ONE HUNDRED AND TWENTY THREE THOUSAND FOUR HUNDRED AND FIFTY SIX ONLY"
    print(formatter.format_string("123456"))  # Should print the same
    print(formatter.trans_two("23"))  # Should print "TWENTY THREE"
    print(formatter.trans_three("123"))  # Should print "ONE HUNDRED AND TWENTY THREE"
    print(formatter.parse_more(1))  # Should print "THOUSAND"