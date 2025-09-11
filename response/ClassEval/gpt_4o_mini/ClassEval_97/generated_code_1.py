class Words2Numbers:
    """
    The class provides a text-to-number conversion utility, allowing conversion of written numbers (in words) to their numerical representation.
    """

    def __init__(self):
        """
        Initialize the word lists and dictionaries required for conversion
        """
        self.numwords = {}
        self.units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        self.tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        self.scales = ["hundred", "thousand", "million", "billion", "trillion"]

        self.numwords["and"] = (1, 0)
        for idx, word in enumerate(self.units):
            self.numwords[word] = (1, idx)
        for idx, word in enumerate(self.tens):
            self.numwords[word] = (1, idx * 10)
        for idx, word in enumerate(self.scales):
            self.numwords[word] = (10 ** (idx * 3 or 2), 0)

        self.ordinal_words = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
        self.ordinal_endings = [('ieth', 'y'), ('th', '')]

    def text2int(self, textnum):
        """
        Convert the word string to the corresponding integer string
        :param textnum: string, the word string to be converted
        :return: string, the final converted integer string
        >>> w2n = Words2Numbers()
        >>> w2n.text2int("thirty-two")
        "32"
        """
        textnum = textnum.replace("-", " ")
        current = result = 0
        for word in textnum.split():
            if word not in self.numwords:
                raise ValueError(f"Invalid input word: {word}")
            scale, increment = self.numwords[word]
            current += increment
            if scale > 1:
                current *= scale
                result += current
                current = 0
        return str(result + current)

    def is_valid_input(self, textnum):
        """
        Check if the input text contains only valid words that can be converted into numbers.
        :param textnum: The input text containing words representing numbers.
        :return: True if input is valid, False otherwise.
        >>> w2n = Words2Numbers()
        >>> w2n.is_valid_input("thirty-two")
        True
        >>> w2n.is_valid_input("forty-three and one")
        True
        >>> w2n.is_valid_input("forty-three and three cats")
        False
        """
        words = textnum.replace("-", " ").split()
        valid_words = set(self.numwords.keys())
        return all(word in valid_words for word in words)

if __name__ == "__main__":
    # Test case for text2int
    w2n = Words2Numbers()
    output_text2int = w2n.text2int("thirty-two")
    print(output_text2int)  # Expected output: "32"

    # Test case for is_valid_input
    output_is_valid_input_1 = w2n.is_valid_input("thirty-two")
    print(output_is_valid_input_1)  # Expected output: True
    output_is_valid_input_2 = w2n.is_valid_input("forty-three and three cats")
    print(output_is_valid_input_2)  # Expected output: False