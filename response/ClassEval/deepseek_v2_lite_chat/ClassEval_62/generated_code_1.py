import re

class NLPDataProcessor:
    """
    The class processes NLP data by removing stop words from a list of strings using a pre-defined stop word list.
    """

    def construct_stop_word_list(self):
        """
        Construct a stop word list including 'a', 'an', 'the'.
        :return: a list of stop words
        >>> NLPDataProcessor.construct_stop_word_list()
        ['a', 'an', 'the']
        """
        return ['a', 'an', 'the']

    def remove_stop_words(self, string_list, stop_word_list):
        """
        Remove all the stop words from the list of strings.
        :param string_list: a list of strings
        :param stop_word_list: a list of stop words
        :return: a list of words without stop words
        """
        words = re.findall(r'\b\w+\b', ' '.join(string_list))
        return [word for word in words if word.casefold() not in map(str.casefold, stop_word_list)]

    def process(self, string_list):
        """
        Construct a stop word list including 'a', 'an', 'the', and remove all the stop words from the list of strings.
        :param string_list: a list of strings
        :return: a list of words without stop words
        """
        stop_word_list = self.construct_stop_word_list()
        return self.remove_stop_words(string_list, stop_word_list)

# Test cases
if __name__ == "__main__":
    processor = NLPDataProcessor()
    test_string_list = ['This is a test.']

    # Test for construct_stop_word_list
    stop_words = processor.construct_stop_word_list()
    print("Constructed Stop Words:", stop_words)  # Expected: ['a', 'an', 'the']

    # Test for remove_stop_words
    filtered_words = processor.remove_stop_words(test_string_list, stop_words)
    print("Filtered Words:", filtered_words)  # Expected: ['This', 'is', 'test.']

    # Test for process
    processed_string_list = processor.process(test_string_list)
    print("Processed String List:", processed_string_list)  # Expected: ['This', 'is', 'test.']