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
        Append the input word into self.word_list.
        :param word: str, input word
        """
        self.word_list.append(word)

    def find_longest_word(self, sentence):
        """
        Remove punctuation marks and split a sentence into a list of word. Find the longest splited word that is in the self.word_list.
        Words are strictly case sensitive.
        :param sentence: a sentence str
        :return: str, the longest splited word that is in the self.word_list, or '' if the self.word_list is empty.
        >>> longestWord = LongestWord()
        >>> longestWord.add_word('A')
        >>> longestWord.add_word('aM')
        >>> longestWord.find_longest_word('I am a student.')
        'a'
        """
        # Remove punctuation marks from the sentence
        sentence = re.sub(r'[^a-zA-Z]', '', sentence)

        # Split the sentence into a list of words
        words = sentence.split()

        # Find the longest word in the list
        longest_word = ''
        for word in words:
            if word in self.word_list:
                if len(word) > len(longest_word):
                    longest_word = word

        # Return the longest word or '' if the self.word_list is empty
        if longest_word:
            return longest_word
        else:
            return ''
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Example usage:
    longestWord = LongestWord()
    longestWord.add_word('A')
    longestWord.add_word('aM')
    print(longestWord.find_longest_word('I am a student.'))  # Output: 'a'
    print(longestWord.find_longest_word('I am a student.'))  # Output: '' if no match found