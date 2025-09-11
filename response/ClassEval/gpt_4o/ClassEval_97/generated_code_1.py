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
        """
        current = result = 0
        for word in textnum.replace('-', ' ').split():
            if word in self.ordinal_words:
                scale, increment = 1, self.ordinal_words[word]
            else:
                # Handle ordinal endings
                for ending, replacement in self.ordinal_endings:
                    if word.endswith(ending):
                        word = word[:-len(ending)] + replacement

                if word not in self.numwords:
                    raise ValueError(f"Invalid word: {word}")

                scale, increment = self.numwords[word]

            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0

        return str(result + current)

    def is_valid_input(self, textnum):
        """
        Check if the input text contains only valid words that can be converted into numbers.
        :param textnum: The input text containing words representing numbers.
        :return: True if input is valid, False otherwise.
        """
        for word in textnum.replace('-', ' ').split():
            # Handle ordinal endings
            for ending, replacement in self.ordinal_endings:
                if word.endswith(ending):
                    word = word[:-len(ending)] + replacement

            if word not in self.numwords and word not in self.ordinal_words:
                return False
        return True


if __name__ == "__main__":
    # Test case for text2int
    w2n = Words2Numbers()
    print(w2n.text2int("thirty-two"))  # Output: "32"

    # Test case for is_valid_input
    print(w2n.is_valid_input("thirty-two"))  # Output: True
    print(w2n.is_valid_input("thirty-second"))  # Output: True
    print(w2n.is_valid_input("invalid-word"))  # Output: False