import re

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
        if textnum.isdigit():
            return textnum
        
        current = result = 0
        tokens = re.findall(r'\d+|\w+', textnum)
        
        for i, token in enumerate(tokens):
            if token.isdigit():
                current += int(token)
            else:
                if token in self.numwords:
                    scale, increment = self.numwords[token]
                    current *= scale
                    current += increment
                elif token in self.ordinal_words:
                    current += self.ordinal_words[token]
                elif token == 'negative':
                    result = -result
                else:
                    raise ValueError(f"Illegal word: {token}")
                
                if token in self.scales:
                    result += current
                    current = 0
        
        result += current
        return str(result)

    def is_valid_input(self, textnum):
        """
        Check if the input text contains only valid words that can be converted into numbers.
        :param textnum: The input text containing words representing numbers.
        :return: True if input is valid, False otherwise.
        """
        tokens = re.findall(r'\d+|\w+', textnum)
        for token in tokens:
            if token.isdigit():
                continue
            elif token in self.numwords or token in self.ordinal_words:
                continue
            else:
                return False
        return True

if __name__ == "__main__":
    w2n = Words2Numbers()
    
    # Test cases
    print(w2n.text2int("thirty-two"))  # Output: "32"
    print(w2n.text2int("one hundred twenty-three"))  # Output: "123"
    print(w2n.text2int("two thousand four hundred fifty-six"))  # Output: "2456"
    print(w2n.text2int("negative one million two hundred thirty-four thousand five hundred sixty-seven"))  # Output: "-1234567"
    
    print(w2n.is_valid_input("thirty-two"))  # Output: True
    print(w2n.is_valid_input("one hundred twenty-three"))  # Output: True
    print(w2n.is_valid_input("two thousand four hundred fifty-six"))  # Output: True
    print(w2n.is_valid_input("negative one million two hundred thirty-four thousand five hundred sixty-seven"))  # Output: True
    print(w2n.is_valid_input("one hundred twenty-three thousand"))  # Output: True
    print(w2n.is_valid_input("one hundred twenty-three thousand five hundred and sixty-seven"))  # Output: True
    print(w2n.is_valid_input("one hundred twenty-three thousand five hundred and sixty-seven million eight hundred ninety"))  # Output: True
    print(w2n.is_valid_input("one hundred twenty-three thousand five hundred and sixty-seven million eight hundred ninety thousand"))  # Output: True
    
    print(w2n.is_valid_input("one hundred twenty-three thousand five hundred and sixty-seven million eight hundred ninety thousand and"))  # Output: False
    print(w2n.is_valid_input("one hundred twenty-three thousand five hundred and sixty-seven million eight hundred ninety thousand one hundred twenty-three"))  # Output: False