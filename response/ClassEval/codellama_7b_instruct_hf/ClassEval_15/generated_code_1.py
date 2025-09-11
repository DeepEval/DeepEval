import unittest

class BoyerMooreSearch:
    def __init__(self, text, pattern):
        self.text = text
        self.pattern = pattern
        self.textLen = len(text)
        self.patLen = len(pattern)

    def match_in_pattern(self, char):
        for i in range(self.patLen):
            if self.pattern[i] == char:
                return i
        return -1

    def mismatch_in_text(self, currentPos):
        for i in range(currentPos, self.textLen):
            if self.text[i] != self.pattern[self.patLen - 1]:
                return i
        return -1

    def bad_character_heuristic(self):
        lastChar = self.pattern[-1]
        for i in range(self.textLen - self.patLen + 1):
            if self.text[i + self.patLen - 1] == lastChar:
                return i
        return -1

class BoyerMooreSearchTest(unittest.TestCase):
    def test_match_in_pattern(self):
        bms = BoyerMooreSearch('ABAABA', 'AB')
        self.assertEqual(bms.match_in_pattern('A'), 0)

    def test_mismatch_in_text(self):
        bms = BoyerMooreSearch('ABAABA', 'ABC')
        self.assertEqual(bms.mismatch_in_text(0), 2)

    def test_bad_character_heuristic(self):
        bms = BoyerMooreSearch('ABAABA', 'AB')
        self.assertEqual(bms.bad_character_heuristic(), [0, 3])

if __name__ == '__main__':
    unittest.main()