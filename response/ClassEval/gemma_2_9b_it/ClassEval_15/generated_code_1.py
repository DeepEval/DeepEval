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

    def match_in_pattern(self, char):
        """
        Finds the rightmost occurrence of a character in the pattern.
        :param char: The character to be searched for, str.
        :return: The index of the rightmost occurrence of the character in the pattern, int.
        >>> boyerMooreSearch = BoyerMooreSearch("ABAABA", "AB")
        >>> boyerMooreSearch.match_in_pattern("A")
        0

        """
        return self.pattern.rfind(char)

    def mismatch_in_text(self, currentPos):
        """
        Determines the position of the first dismatch between the pattern and the text.
        :param currentPos: The current position in the text, int.
        :return: The position of the first dismatch between the pattern and the text, int,otherwise -1.
        >>> boyerMooreSearch = BoyerMooreSearch("ABAABA", "ABC")
        >>> boyerMooreSearch.mismatch_in_text(0)
        2

        """
        i = self.patLen - 1
        while i >= 0 and self.text[currentPos + i] == self.pattern[i]:
            i -= 1
        if i == -1:
            return -1
        return self.patLen - 1 - i

    def bad_character_heuristic(self):
        """
        Finds all occurrences of the pattern in the text.
        :return: A list of all positions of the pattern in the text, list.
        >>> boyerMooreSearch = BoyerMooreSearch("ABAABA", "AB")
        >>> boyerMooreSearch.bad_character_heuristic()
        [0, 3]

        """
        occurrences = []
        i = self.patLen - 1
        while i < self.textLen:
            j = self.mismatch_in_text(i)
            if j == -1:
                occurrences.append(i - self.patLen + 1)
                i += self.patLen
            else:
                shift = self.patLen - j - 1
                i += max(1, shift)
        return occurrences

if __name__ == "__main__":
    text = "ABAABA"
    pattern = "AB"
    boyerMooreSearch = BoyerMooreSearch(text, pattern)
    print(boyerMooreSearch.bad_character_heuristic())