class NLPDataProcessor:
    """
    The class processes NLP data by removing stop words from a list of strings using a pre-defined stop word list.
    """

    def construct_stop_word_list(self):
        """
        Construct a stop word list including 'a', 'an', 'the'.
        :return: a list of stop words
        """
        return ['a', 'an', 'the']

    def remove_stop_words(self, string_list, stop_word_list):
        """
        Remove all the stop words from the list of strings.
        :param string_list: a list of strings
        :param stop_word_list: a list of stop words
        :return: a list of words without stop words
        """
        processed_list = []
        for string in string_list:
            words = string.split()
            filtered_words = [word for word in words if word.lower() not in stop_word_list]
            processed_list.append(filtered_words)
        return processed_list

    def process(self, string_list):
        """
        Construct a stop word list including 'a', 'an', 'the', and remove all the stop words from the list of strings.
        :param string_list: a list of strings
        :return: a list of words without stop words
        """
        stop_word_list = self.construct_stop_word_list()
        return self.remove_stop_words(string_list, stop_word_list)


if __name__ == "__main__":
    nlp_processor = NLPDataProcessor()

    # Test cases
    print("Testing construct_stop_word_list:")
    print(nlp_processor.construct_stop_word_list())  # Expected output: ['a', 'an', 'the']

    print("\nTesting remove_stop_words:")
    test_strings = ['This is a test.', 'This is an example.']
    print(nlp_processor.remove_stop_words(test_strings, nlp_processor.construct_stop_word_list()))
    # Expected output: [['This', 'is', 'test.'], ['This', 'is', 'example.']]

    print("\nTesting process:")
    print(nlp_processor.process(['This is a test.']))  # Expected output: [['This', 'is', 'test.']]