class NLPDataProcessor:
    """
    The class processes NLP data by removing stop words from a list of strings using a pre-defined stop word list.
    """

    @staticmethod
    def construct_stop_word_list():
        """
        Construct a stop word list including 'a', 'an', 'the'.
        :return: a list of stop words
        >>> NLPDataProcessor.construct_stop_word_list()
        ['a', 'an', 'the']
        """
        return ['a', 'an', 'the']

    @staticmethod
    def remove_stop_words(string_list, stop_word_list):
        """
        Remove all the stop words from the list of strings.
        :param string_list: a list of strings
        :param stop_word_list: a list of stop words
        :return: a list of words without stop words
        >>> NLPDataProcessor.remove_stop_words(['This is a test.'], NLPDataProcessor.construct_stop_word_list())
        [['This', 'is', 'test.']]
        """
        # Split each string into words and filter out stop words
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
        >>> NLPDataProcessor().process(['This is a test.'])
        [['This', 'is', 'test.']]
        """
        stop_word_list = self.construct_stop_word_list()
        return self.remove_stop_words(string_list, stop_word_list)

# Test cases
if __name__ == "__main__":
    # Test case for construct_stop_word_list
    stop_words = NLPDataProcessor.construct_stop_word_list()
    print("Stop words:", stop_words)

    # Test case for remove_stop_words
    removed_stop_words = NLPDataProcessor.remove_stop_words(['This is a test.'], NLPDataProcessor.construct_stop_word_list())
    print("Removed stop words:", removed_stop_words)

    # Test case for process
    processed_data = NLPDataProcessor().process(['This is a test.'])
    print("Processed data:", processed_data)