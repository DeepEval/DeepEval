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
        textnum = textnum.lower()
        current = result = 0
        curstring = ""
        on_num = on_scale = False
        for word in textnum.split():
            if word in self.numwords:
                scale, increment = self.numwords[word]
                curstring += word
                if on_num:
                    if scale == 1 and increment > 0:
                        result += current * scale
                        current = increment
                    else:
                        current *= scale
                        current += increment
                else:
                    current = increment
                    on_num = True
                on_scale = (scale != 1)
            else:
                if word in self.ordinal_words:
                    result += self.ordinal_words[word]
                elif word in self.ordinal_endings:
                    curstring = curstring[:-1] + self.ordinal_endings[word]
                else:
                    return "Invalid input"
        if on_num:
            result += current
        return str(result)

    def is_valid_input(self, textnum):
        """
        Check if the input text contains only valid words that can be converted into numbers.
        :param textnum: The input text containing words representing numbers.
        :return: True if input is valid, False otherwise.
        >>> w2n = Words2Numbers()
        >>> w2n.is_valid_input("thirty-two")
        False
        """
        textnum = textnum.lower()
        for word in textnum.split():
            if word not in self.numwords and word not in self.ordinal_words and word not in self.ordinal_endings:
                return False
        return True


if __name__ == "__main__":
    w2n = Words2Numbers()
    textnum = "thirty-two"
    
    if w2n.is_valid_input(textnum):
        output = w2n.text2int(textnum)
        print(output)
    else:
        print("Invalid input")