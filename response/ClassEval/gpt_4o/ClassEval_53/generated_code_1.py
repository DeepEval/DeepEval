import re
import string

class LongestWord:
    """
    This class allows adding words to a list and finding the longest word in a given sentence by comparing the words with the ones in the word list.
    """

    def __init__(self):
        """
        Initialize a list of words.
        """
        self.word_list = []

    def add_word(self, word):
        """
        Append the input word into self.word_list.
        :param word: str, input word
        """
        self.word_list.append(word)

    def find_longest_word(self, sentence):
        """
        Remove punctuation marks and split a sentence into a list of words. Find the longest split word that is in the self.word_list.
        Words are strictly case sensitive.
        :param sentence: a sentence str
        :return str: longest split word that is in the self.word_list. return '' if self.word_list is empty.
        """
        # Remove punctuation from the sentence
        sentence = re.sub(r'[' + string.punctuation + ']', '', sentence)
        
        # Split the sentence into words
        words = sentence.split()
        
        # Find the longest word in self.word_list that is also in the sentence words
        longest_word = ''
        for word in words:
            if word in self.word_list and len(word) > len(longest_word):
                longest_word = word
        
        return longest_word

if __name__ == "__main__":
    # Create an instance of LongestWord
    longestWord = LongestWord()
    
    # Test case for add_word method
    longestWord.add_word('A')
    longestWord.add_word('aM')
    
    # Test case for find_longest_word method
    output = longestWord.find_longest_word('I am a student.')
    print(output)  # Expected output: 'a'