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
                            "EIGHTEEN", "NINETEEN"]
        self.NUMBER_TEN = ["", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
        self.NUMBER_MORE = ["", "THOUSAND", "MILLION", "BILLION"]
        self.NUMBER_SUFFIX = ["k", "w", "", "m", "", "", "b", "", "", "t", "", "", "p", "", "", "e"]

    def format(self, x):
        """
        Converts a number into words format
        :param x: int or float, the number to be converted into words format
        :return: str, the number in words format
        """
        if isinstance(x, float):
            whole, decimal = str(x).split(".")
            return f"{self.format_string(whole)} POINT {self.format_string(decimal)}"
        else:
            return self.format_string(str(x))

    def format_string(self, x):
        """
        Converts a string representation of a number into words format
        :param x: str, the string representation of a number
        :return: str, the number in words format
        """
        n = int(x)
        if n == 0:
            return "ZERO ONLY"
        
        words = []
        group = 0
        while n > 0:
            part = n % 1000
            if part > 0:
                words.append(self.trans_three(str(part)))
                words.append(self.parse_more(group))
            n //= 1000
            group += 1

        words = [word for word in words if word]
        return " ".join(words[::-1]).strip() + " ONLY"

    def trans_two(self, s):
        """
        Converts a two-digit number into words format
        :param s: str, the two-digit number
        :return: str, the number in words format
        """
        n = int(s)
        if n < 10:
            return self.NUMBER[n]
        elif n < 20:
            return self.NUMBER_TEEN[n - 10]
        else:
            tens = n // 10
            ones = n % 10
            return self.NUMBER_TEN[tens] + (" " + self.NUMBER[ones] if ones > 0 else "")

    def trans_three(self, s):
        """
        Converts a three-digit number into words format
        :param s: str, the three-digit number
        :return: str, the number in words format
        """
        n = int(s)
        hundreds = n // 100
        remainder = n % 100
        if hundreds > 0 and remainder > 0:
            return f"{self.NUMBER[hundreds]} HUNDRED AND {self.trans_two(str(remainder).zfill(2))}"
        elif hundreds > 0:
            return f"{self.NUMBER[hundreds]} HUNDRED"
        else:
            return self.trans_two(str(remainder).zfill(2))

    def parse_more(self, i):
        """
        Parses the thousand/million/billion suffix based on the index
        :param i: int, the index representing the magnitude (thousand, million, billion)
        :return: str, the corresponding suffix for the magnitude
        """
        return self.NUMBER_MORE[i]

if __name__ == "__main__":
    formatter = NumberWordFormatter()
    
    # Test case for format
    output = formatter.format(123456)
    print(output)  # Expected: "ONE HUNDRED AND TWENTY THREE THOUSAND FOUR HUNDRED AND FIFTY SIX ONLY"

    # Test case for format_string
    output = formatter.format_string("123456")
    print(output)  # Expected: "ONE HUNDRED AND TWENTY THREE THOUSAND FOUR HUNDRED AND FIFTY SIX ONLY"

    # Test case for trans_two
    output = formatter.trans_two("23")
    print(output)  # Expected: "TWENTY THREE"

    # Test case for trans_three
    output = formatter.trans_three("123")
    print(output)  # Expected: "ONE HUNDRED AND TWENTY THREE"

    # Test case for parse_more
    output = formatter.parse_more(1)
    print(output)  # Expected: "THOUSAND"