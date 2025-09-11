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
        """
        for i in range(self.patLen - 1, -1, -1):
            if self.pattern[i] == char:
                return i
        return -1

    def mismatch_in_text(self, currentPos):
        """
        Determines the position of the first mismatch between the pattern and the text.
        :param currentPos: The current position in the text, int.
        :return: The position of the first mismatch between the pattern and the text, int, otherwise -1.
        """
        for i in range(self.patLen):
            if currentPos + i >= self.textLen or self.text[currentPos + i] != self.pattern[i]:
                return i
        return -1

    def bad_character_heuristic(self):
        """
        Finds all occurrences of the pattern in the text.
        :return: A list of all positions of the pattern in the text, list.
        """
        positions = []
        shift = 0

        while shift <= self.textLen - self.patLen:
            mismatchPos = self.mismatch_in_text(shift)
            if mismatchPos == -1:
                positions.append(shift)
                shift += 1
            else:
                char = self.text[shift + mismatchPos]
                badCharShift = mismatchPos - self.match_in_pattern(char)
                shift += max(1, badCharShift)

        return positions

if __name__ == "__main__":
    # Test match_in_pattern
    boyerMooreSearch = BoyerMooreSearch("ABAABA", "AB")
    output = boyerMooreSearch.match_in_pattern("A")
    print(output)  # Expected: 0

    # Test mismatch_in_text
    boyerMooreSearch = BoyerMooreSearch("ABAABA", "ABC")
    output = boyerMooreSearch.mismatch_in_text(0)
    print(output)  # Expected: 2

    # Test bad_character_heuristic
    boyerMooreSearch = BoyerMooreSearch("ABAABA", "AB")
    output = boyerMooreSearch.bad_character_heuristic()
    print(output)  # Expected: [0, 3]