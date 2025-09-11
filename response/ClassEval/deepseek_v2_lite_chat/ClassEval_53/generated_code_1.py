import re
import string

class LongestWord:
    """
    This is a class that allows to add words to a list and find the longest word in a given sentence by comparing the words with the ones in the word list.
    """

    def __init__(self):
        """
        Initialize a list of word.
        """
        self.word_list = []

    def add_word(self, word):
        """
        Append the input word into self.word_list
        :param word: str, input word
        """
        self.word_list.append(word.upper())  # Convert to upper case before appending

    def find_longest_word(self):
        """
        Remove punctuation marks and split a sentence into a list of words. Find the longest splited word that is in the self.word_list.
        Words are strictly case sensitive.
        :param sentence: a sentence str
        :return str: longest splitted word that is in the self.word_list. Return '' if self.word_list is empty.
        """
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', sentence.upper())  # Convert to upper case to ensure case sensitivity
        if not self.word_list:
            return ''
        longest_word = max(self.word_list, key=len)  # Find the longest word in the list
        for word in words:
            if word == longest_word:
                return word  # Return the longest word if it matches
        return ''  # Return an empty string if no match is found

# Test cases
if __name__ == "__main__":
    longestWord = LongestWord()
    longestWord.add_word('A')
    longestWord.add_word('aM')
    print(longestWord.find_longest_word('I am a student.'))  # Expected output: 'a'