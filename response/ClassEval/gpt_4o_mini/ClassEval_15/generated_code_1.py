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
        >>> boyerMooreSearch.match_in_pattern("B")
        1
        >>> boyerMooreSearch.match_in_pattern("C")
        -1
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
        >>> boyerMooreSearch = BoyerMooreSearch("ABAABA", "ABC")
        >>> boyerMooreSearch.mismatch_in_text(0)
        2
        >>> boyerMooreSearch.mismatch_in_text(1)
        1
        >>> boyerMooreSearch.mismatch_in_text(3)
        -1
        """
        for i in range(self.patLen):
            if currentPos + i >= self.textLen or self.text[currentPos + i] != self.pattern[i]:
                return i
        return -1

    def bad_character_heuristic(self):
        """
        Finds all occurrences of the pattern in the text.
        :return: A list of all positions of the pattern in the text, list.
        >>> boyerMooreSearch = BoyerMooreSearch("ABAABA", "AB")
        >>> boyerMooreSearch.bad_character_heuristic()
        [0, 3]
        >>> boyerMooreSearch2 = BoyerMooreSearch("AAABAAA", "A")
        >>> boyerMooreSearch2.bad_character_heuristic()
        [0, 1, 2, 4, 5, 6]
        """
        occurrences = []
        skip = 0
        
        while skip <= self.textLen - self.patLen:
            mismatch_index = self.mismatch_in_text(skip)
            if mismatch_index == -1:
                occurrences.append(skip)
                skip += self.patLen - self.match_in_pattern(self.text[skip + self.patLen - 1]) if skip + self.patLen - 1 < self.textLen else 1
            else:
                skip += max(1, mismatch_index - self.match_in_pattern(self.text[skip + mismatch_index]) if skip + mismatch_index < self.textLen else -1)

        return occurrences


if __name__ == "__main__":
    # Testing the match_in_pattern method
    boyerMooreSearch = BoyerMooreSearch("ABAABA", "AB")
    print(boyerMooreSearch.match_in_pattern("A"))  # Output: 0
    print(boyerMooreSearch.match_in_pattern("B"))  # Output: 1
    print(boyerMooreSearch.match_in_pattern("C"))  # Output: -1

    # Testing the mismatch_in_text method
    boyerMooreSearch = BoyerMooreSearch("ABAABA", "ABC")
    print(boyerMooreSearch.mismatch_in_text(0))  # Output: 2
    print(boyerMooreSearch.mismatch_in_text(1))  # Output: 1
    print(boyerMooreSearch.mismatch_in_text(3))  # Output: -1

    # Testing the bad_character_heuristic method
    boyerMooreSearch = BoyerMooreSearch("ABAABA", "AB")
    print(boyerMooreSearch.bad_character_heuristic())  # Output: [0, 3]
    boyerMooreSearch2 = BoyerMooreSearch("AAABAAA", "A")
    print(boyerMooreSearch2.bad_character_heuristic())  # Output: [0, 1, 2, 4, 5, 6]