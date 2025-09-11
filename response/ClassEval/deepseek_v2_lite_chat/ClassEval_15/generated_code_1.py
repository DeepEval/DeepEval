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
        self.text = text
        self.pattern = pattern
        self.textLen = len(text)
        self.patLen = len(pattern)

    def match_in_pattern(self, char):
        """
        Finds the rightmost occurrence of a character in the pattern.
        :param char: The character to be searched for, str.
        :return: The index of the rightmost occurrence of the character in the pattern, int.
        """
        if char in self.pattern:
            return self.pattern.rindex(char)
        else:
            return -1

    def mismatch_in_text(self, currentPos):
        """
        Determines the position of the first dismatch between the pattern and the text.
        :param currentPos: The current position in the text, int.
        :return: The position of the first dismatch between the pattern and the text, int, otherwise -1.
        """
        last_occurence = self.patLen - 1
        for i in range(last_occurence, currentPos):
            if self.pattern[i:i+self.patLen] == self.text[i:i+self.patLen]:
                return i
        return -1

    def bad_character_heuristic(self):
        """
        Finds all occurrences of the pattern in the text.
        :return: A list of all positions of the pattern in the text, list.
        """
        occurrences = []
        for i in range(self.textLen - self.patLen + 1):
            if self.pattern == self.text[i:i+self.patLen]:
                occurrences.append(i)
        return occurrences

# Example usage:
if __name__ == "__main__":
    boyerMooreSearch = BoyerMooreSearch("ABAABA", "AB")
    print(boyerMooreSearch.match_in_pattern("A"))  # Should return 0
    print(boyerMooreSearch.mismatch_in_text(0))      # Should return 2
    print(boyerMooreSearch.bad_character_heuristic()) # Should return [0, 3]