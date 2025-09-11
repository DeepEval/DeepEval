import re
from collections import Counter

class NLPDataProcessor2:
    """
    The class processes NLP data by extracting words from a list of strings, calculating the frequency of each word, and returning the top 5 most frequent words.
    """

    def process_data(self, string_list):
        """
        Keep only English letters and spaces in the string, convert the string to lower case, and split the string into a list of words.
        :param string_list: a list of strings
        :return: words_list: a list of words lists
        >>> NLPDataProcessor2().process_data(['This is a test.'])
        [['this', 'is', 'a', 'test']]
        """
        words_list = []
        for string in string_list:
            clean_string = re.sub(r'[^a-zA-Z\s]', '', string)
            words = clean_string.lower().split()
            words_list.append(words)
        return words_list

    def calculate_word_frequency(self, words_list):
        """
        Calculate the word frequency of each word in the list of words list, and sort the word frequency dictionary by value in descending order.
        :param words_list: a list of words lists
        :return: top 5 word frequency dictionary, a dictionary of word frequency, key is word, value is frequency
        >>> NLPDataProcessor2().calculate_word_frequency([['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']])
        {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}
        """
        all_words = [word for sublist in words_list for word in sublist]
        frequency = Counter(all_words)
        top_5 = dict(frequency.most_common(5))
        return top_5

    def process(self, string_list):
        """
        Keep only English letters and spaces in the string, convert the string to lower case, and split the string into a list of words. Calculate the word frequency of each word in the list of words list, and sort the word frequency dictionary by value in descending order.
        :param string_list: a list of strings
        :return: top 5 word frequency dictionary, a dictionary of word frequency, key is word, value is frequency
        >>> NLPDataProcessor2().process(['This is a test.', 'This is another test.'])
        {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}
        """
        words_list = self.process_data(string_list)
        return self.calculate_word_frequency(words_list)

if __name__ == "__main__":
    processor = NLPDataProcessor2()
    
    # Test case for process_data method
    output_data = processor.process_data(['This is a test.'])
    print(output_data)  # [['this', 'is', 'a', 'test']]

    # Test case for calculate_word_frequency method
    output_frequency = processor.calculate_word_frequency([['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']])
    print(output_frequency)  # {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}

    # Test case for process method
    output_process = processor.process(['This is a test.', 'This is another test.'])
    print(output_process)  # {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}