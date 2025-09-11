import re
import string

class LongestWord:
    """
    This is a class that allows adding words to a list and finding the longest word in a given sentence by comparing the words with the ones in the word list.
    """

    def __init__(self):
        """
        Initialize a list of words.
        """
        self.word_list = []

    def add_word(self, word):
        """
        Append the input word into self.word_list
        :param word: str, input word
        """
        self.word_list.append(word)

    def find_longest_word(self, sentence):
        """
        Remove punctuation marks and split a sentence into a list of words. Find the longest split word that is in the self.word_list.
        Words are strictly case sensitive.
        :param sentence: a sentence str
        :return str: longest split word that is in the self.word_list. return '' if self.word_list is empty.
        >>> longestWord = LongestWord()
        >>> longestWord.add_word('A')
        >>> longestWord.add_word('aM')
        >>> longestWord.find_longest_word('I am a student.')
        'a'
        """
        if not self.word_list:
            return ''
        
        # Remove punctuation from the sentence
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # Split the sentence into words
        words = sentence.split()
        
        # Find the longest word in self.word_list that is present in the sentence
        longest_word = ''
        for word in words:
            if word in self.word_list and len(word) > len(longest_word):
                longest_word = word
        
        return longest_word

# Test cases
if __name__ == "__main__":
    # Instance of LongestWord
    longestWord = LongestWord()
    
    # Test add_word method
    longestWord.add_word('A')
    longestWord.add_word('aM')
    print(longestWord.word_list)  # Output: ['A', 'aM']
    
    # Test find_longest_word method
    output = longestWord.find_longest_word('I am a student.')
    print(output)  # Output: 'a'

    # Additional test cases
    longestWord.add_word('student')
    output = longestWord.find_longest_word('I am a student. A student is here.')
    print(output)  # Output: 'student'

    longestWord.add_word('the')
    output = longestWord.find_longest_word('I am the best student.')
    print(output)  # Output: 'the'
    
    output = longestWord.find_longest_word('Hello world!')
    print(output)  # Output: ''