class BoyerMooreSearch:
    """
    This is a class that implements the Boyer-Moore algorithm for string searching, which is used to find occurrences of a pattern within a given text.
    """

    def __init__(self, text, pattern):
        """
        Initializes the BoyerMooreSearch class with the given text and pattern.
        :param text: The text to be searched, str.
        :param pattern: The pattern to be searched for, str.
        """
        self.text, self.pattern = text, pattern
        self.textLen, self.patLen = len(text), len(pattern)
        self.bad_char_table = self.bad_character_heuristic()

    def match_in_pattern(self, char):
        """
        Finds the rightmost occurrence of a character in the pattern.
        :param char: The character to be searched for, str.
        :return: The index of the rightmost occurrence of the character in the pattern, int.
        """
        return self.pattern.rfind(char)

    def mismatch_in_text(self, currentPos):
        """
        Determines the position of the first dismatch between the pattern and the text.
        :param currentPos: The current position in the text, int.
        :return: The position of the first dismatch between the pattern and the text, int,otherwise -1.
        """
        for i in range(self.patLen - 1, -1, -1):
            if self.pattern[i] != self.text[currentPos + i]:
                return currentPos + i
        return -1

    def bad_character_heuristic(self):
        """
        Finds all occurrences of the pattern in the text.
        :return: A list of all positions of the pattern in the text, list.
        """
        table = {}
        for i in range(self.patLen):
            table[self.pattern[i]] = i
        return table

    def bad_character_heuristic(self):
        """
        Finds all occurrences of the pattern in the text.
        :return: A list of all positions of the pattern in the text, list.
        """
        table = {}
        for i in range(self.patLen):
            table[self.pattern[i]] = i
        return table

    def search(self):
        """
        Finds all occurrences of the pattern in the text.
        :return: A list of all positions of the pattern in the text, list.
        """
        occurrences = []
        i = 0
        while i <= self.textLen - self.patLen:
            j = self.patLen - 1

            # Keep reducing j while characters of the pattern and text are matching
            while j >= 0 and self.pattern[j] == self.text[i + j]:
                j -= 1

            if j < 0:
                occurrences.append(i)
                i += (self.patLen - self.bad_char_table.get(self.text[i + self.patLen], -1) - 1) if i + self.patLen < self.textLen else 1
            else:
                i += max(1, j - self.bad_char_table.get(self.text[i + j], -1))
        return occurrences

if __name__ == "__main__":
    boyerMooreSearch = BoyerMooreSearch("ABAABA", "AB")
    print("Test match_in_pattern:")
    print(boyerMooreSearch.match_in_pattern("A"))  # Output: 0

    print("\nTest mismatch_in_text:")
    boyerMooreSearch = BoyerMooreSearch("ABAABA", "ABC")
    print(boyerMooreSearch.mismatch_in_text(0))  # Output: 2

    print("\nTest bad_character_heuristic:")
    boyerMooreSearch = BoyerMooreSearch("ABAABA", "AB")
    print(boyerMooreSearch.bad_character_heuristic())  # Output: {'A': 0, 'B': 1}

    print("\nTest search:")
    boyerMooreSearch = BoyerMooreSearch("ABAABA", "AB")
    print(boyerMooreSearch.search())  # Output: [0, 3]