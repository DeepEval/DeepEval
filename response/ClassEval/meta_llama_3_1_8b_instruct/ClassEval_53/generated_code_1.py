import re
import string

class LongestWord:
    """
    This is a class allows to add words to a list and find the longest word in a given sentence by comparing the words with the ones in the word list.
    """

    def __init__(self):
        """
        Initialize a list of word.
        """
        self.word_list = []

    def add_word(self, word):
        """
        append the input word into self.word_list
        :param word: str, input word
        """
        # Add the word to the list
        self.word_list.append(word)
        return f"Word '{word}' added to the list."

    def find_longest_word(self, sentence):
        """
        Remove punctuation marks and split a sentence into a list of word. Find the longest splited word that is in the self.word_list.
        Words are strictly case sensitive.
        :param sentence: a sentence str
        :return str: longest splited word that is in the self.word_list. return '' if self.word_list is empty.
        """
        # Remove punctuation marks from the sentence
        sentence = re.sub('['+string.punctuation+']', '', sentence)
        
        # Split the sentence into words
        words = sentence.split()
        
        # Find the longest word in the sentence that is in the word list
        longest_word = max((word for word in words if word in self.word_list), key=len, default='')
        
        return longest_word


if __name__ == "__main__":
    longestWord = LongestWord()
    
    # Test case for add_word method
    print(longestWord.add_word('A'))  # Output: Word 'A' added to the list.
    print(longestWord.add_word('aM'))  # Output: Word 'aM' added to the list.
    
    # Test case for find_longest_word method
    print(longestWord.find_longest_word('I am a student.'))  # Output: a
    print(longestWord.find_longest_word('I am an awesome student.'))  # Output: a