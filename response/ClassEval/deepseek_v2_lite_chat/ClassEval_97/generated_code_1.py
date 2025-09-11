class Words2Numbers:
    def __init__(self):
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
        # Convert ordinal numbers
        ordinal = 'th'
        for word, value in self.ordinal_words.items():
            if textnum.lower().endswith(word.lower() + ordinal):
                return str(value[0])

        # Convert hundreds and thousands
        for word in self.ordinal_words:
            if textnum.lower().startswith(word.lower()):
                return self.convert_hundred_or_thousand(textnum.replace(word, ''), value=(100 if word == 'hundred' else 1000))

        # Convert other numbers
        result = 0
        words = textnum.split()
        for idx, word in enumerate(words):
            if idx > 0 and word.isdigit():
                result += int(word) * (10 ** (idx * 3 or 2))
            elif word in self.numwords:
                result += self.numwords[word][0]
        return str(result)

    def is_valid_input(self, textnum):
        """
        Check if the input text contains only valid words that can be converted into numbers.
        :param textnum: The input text containing words representing numbers.
        :return: True if input is valid, False otherwise.
        """
        return all(word in self.numwords for word in textnum.split())


if __name__ == "__main__":
    w2n = Words2Numbers()

    # Test case 1: Valid input
    print(w2n.text2int("thirty-two"))  # Should return "32"

    # Test case 2: Ordinal number
    print(w2n.text2int("twelfth"))  # Should return "12"

    # Test case 3: Invalid input
    print(w2n.text2int("twenty-one hundred"))  # Should return False

    # Test case 4: Non-numeric characters
    print(w2n.text2int("thirteen-thousand-one hundred"))  # Should return False

    # Test case 5: Invalid word
    print(w2n.text2int("thirty-five-hundred"))  # Should return False

    # Test case 6: Valid input with spaces
    print(w2n.text2int("thirty two"))  # Should return "32"

    # Test case 7: Valid input with hyphens
    print(w2n.text2int("three-hundred-fifty-thousand-two-hundred-and-fifty"))  # Should return "3500200"

    # Test case 8: Invalid ordinal number without the correct ending
    print(w2n.text2int("ninth"))  # Should return False

    # Test case 9: Invalid ordinal number with the wrong ending
    print(w2n.text2int("first"))  # Should return False

    # Test case 10: Invalid ordinal number with no ending
    print(w2n.text2int("firstieth"))  # Should return False

    # Test case 11: Ordinal number with the correct ending
    print(w2n.text2int("eighth"))  # Should return "8"

    # Test case 12: Ordinal number with the correct ending and scale
    print(w2n.text2int("millionth"))  # Should return "1000000"

    # Test case 13: Ordinal number with the correct ending and a zero scale
    print(w2n.text2int("thousandth"))  # Should return "1000"

    # Test case 14: Conversion with hundreds and thousands
    print(w2n.text2int("hundred thousand two thousand fifty"))  # Should return "102050"

    # Test case 15: Conversion with hundreds and thousands in mixed order
    print(w2n.text2int("two thousand fifty hundred thousand"))  # Should return "2050100"