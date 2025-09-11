import re
from collections import Counter

class NLPDataProcessor2:
    """
    The class processes NLP data by extracting words from a list of strings, calculating the frequency of each word, and returning the top 5 most frequent words.
    """


    def process_data(self, string_list):
        """
        keep only English letters and spaces in the string, then convert the string to lower case, and then split the string into a list of words.
        :param string_list: a list of strings
        :return: words_list: a list of words lists
        >>> NLPDataProcessor.process_data(['This is a test.'])
        [['this', 'is', 'a', 'test']]
        """
        words_list = []
        for string in string_list:
            # Keep only English letters and spaces in the string
            string = re.sub(r'[^a-zA-Z ]', '', string)
            # Convert the string to lower case
            string = string.lower()
            # Split the string into a list of words
            words = string.split()
            words_list.append(words)
        return words_list

    def calculate_word_frequency(self, words_list):
        """
        Calculate the word frequency of each word in the list of words list, and sort the word frequency dictionary by value in descending order.
        :param words_list: a list of words lists
        :return: top 5 word frequency dictionary, a dictionary of word frequency, key is word, value is frequency
        >>> NLPDataProcessor.calculate_word_frequency([['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']])
        {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}
        """
        word_frequency = Counter()
        for words in words_list:
            word_frequency.update(words)
        # Sort the word frequency dictionary by value in descending order
        top_5_word_frequency = dict(sorted(word_frequency.items(), key=lambda item: item[1], reverse=True)[:5])
        return top_5_word_frequency, word_frequency

    def process(self, string_list):
        """
        keep only English letters and spaces in the string, then convert the string to lower case, and then split the string into a list of words. Calculate the word frequency of each word in the list of words list, and sort the word frequency dictionary by value in descending order.
        :param string_list: a list of strings
        :return: top 5 word frequency dictionary, a dictionary of word frequency, key is word, value is frequency
        >>> NLPDataProcessor.process(['This is a test.', 'This is another test.'])
        {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}
        """
        words_list = self.process_data(string_list)
        top_5_word_frequency, word_frequency = self.calculate_word_frequency(words_list)
        return top_5_word_frequency, word_frequency

if __name__ == "__main__":
    instance = NLPDataProcessor2()
    # Test case
    string_list = ['This is a test.', 'This is another test.']
    top_5_word_frequency, word_frequency = instance.process(string_list)
    print(top_5_word_frequency)
    print(word_frequency)