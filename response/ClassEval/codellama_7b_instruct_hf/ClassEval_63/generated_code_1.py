import re
from collections import Counter

class NLPDataProcessor:
    def process_data(self, string_list):
        """
        keep only English letters and spaces in the string, then convert the string to lower case, and then split the string into a list of words.
        :param string_list: a list of strings
        :return: words_list: a list of words lists
        """
        words_list = []
        for string in string_list:
            words = re.findall(r'\w+', string.lower())
            words_list.append(words)
        return words_list

    def calculate_word_frequency(self, words_list):
        """
        Calculate the word frequency of each word in the list of words list, and sort the word frequency dictionary by value in descending order.
        :param words_list: a list of words lists
        :return: top 5 word frequency dictionary, a dictionary of word frequency, key is word, value is frequency
        """
        word_frequency = Counter()
        for words in words_list:
            for word in words:
                word_frequency[word] += 1
        return dict(word_frequency.most_common(5))

    def process(self, string_list):
        """
        keep only English letters and spaces in the string, then convert the string to lower case, and then split the string into a list of words. Calculate the word frequency of each word in the list of words list, and sort the word frequency dictionary by value in descending order.
        :param string_list: a list of strings
        :return: top 5 word frequency dictionary, a dictionary of word frequency, key is word, value is frequency
        """
        words_list = self.process_data(string_list)
        return self.calculate_word_frequency(words_list)

if __name__ == "__main__":
    # Test case 1:
    string_list = ["This is a test."]
    nlp_data_processor = NLPDataProcessor()
    words_list = nlp_data_processor.process_data(string_list)
    assert words_list == [['this', 'is', 'a', 'test']]

    # Test case 2:
    string_list = ["This is a test.", "This is another test."]
    word_frequency = nlp_data_processor.calculate_word_frequency(string_list)
    assert word_frequency == {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}

    # Test case 3:
    string_list = ["This is a test.", "This is another test."]
    word_frequency = nlp_data_processor.process(string_list)
    assert word_frequency == {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}