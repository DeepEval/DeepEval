class BoyerMooreSearch:
    """
    This is a class that implements the Boyer-Moore algorithm for string searching, 
    which is used to find occurrences of a pattern within a given text.
    """

    def __init__(self, text, pattern):
        """
        Initializes the BoyerMooreSearch class with the given text and pattern.
        :param text: The text to be searched, str.
        :param pattern: The pattern to be searched for, str.
        """
        self.text, self.pattern = text, pattern
        self.textLen, self.patLen = len(text), len(pattern)
        self.bad_char_table = self._create_bad_char_table()

    def _create_bad_char_table(self):
        """
        Creates a bad character table, which stores the rightmost occurrence of each character in the pattern.
        :return: A dictionary where keys are characters and values are their rightmost occurrences in the pattern, dict.
        """
        bad_char_table = {}
        for i in range(self.patLen):
            if self.pattern[i] not in bad_char_table:
                bad_char_table[self.pattern[i]] = self.patLen - i - 1
        return bad_char_table

    def match_in_pattern(self, char):
        """
        Finds the rightmost occurrence of a character in the pattern.
        :param char: The character to be searched for, str.
        :return: The index of the rightmost occurrence of the character in the pattern, int.
        """
        return self.bad_char_table.get(char, -1)

    def mismatch_in_text(self, currentPos):
        """
        Determines the position of the first dismatch between the pattern and the text.
        :param currentPos: The current position in the text, int.
        :return: The position of the first dismatch between the pattern and the text, int, otherwise -1.
        """
        for i in range(self.patLen - 1, -1, -1):
            if self.text[currentPos + i]!= self.pattern[i]:
                if self.text[currentPos + i] not in self.bad_char_table:
                    return i + 1
                else:
                    shift = i + 1 + self.bad_char_table[self.text[currentPos + i]]
                    if currentPos + i + 1 < self.textLen and shift > i + 1:
                        return i + 1
        return -1

    def bad_character_heuristic(self):
        """
        Finds all occurrences of the pattern in the text using the bad character heuristic.
        :return: A list of all positions of the pattern in the text, list.
        """
        occurrences = []
        shift = 0
        while shift <= self.textLen - self.patLen:
            mismatch_pos = self.mismatch_in_text(shift)
            if mismatch_pos == -1:
                occurrences.append(shift)
            else:
                shift += mismatch_pos
        return occurrences


if __name__ == "__main__":
    instance = BoyerMooreSearch("ABAABA", "AB")
    print(instance.match_in_pattern("A"))  # Output: 0
    print(instance.mismatch_in_text(0))  # Output: 2
    # print(instance.bad_character_heuristic())  # Output: [0, 3]
    
    instance = BoyerMooreSearch("ABAABA", "ABC")
    print(instance.match_in_pattern("C"))  # Output: -1
    print(instance.mismatch_in_text(0))  # Output: 2
    # print(instance.bad_character_heuristic())  # Output: []