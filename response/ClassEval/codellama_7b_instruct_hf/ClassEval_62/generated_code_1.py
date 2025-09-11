import re

class NLPDataProcessor:
    """
    The class processes NLP data by removing stop words from a list of strings using a pre-defined stop word list.
    """

    def __init__(self):
        self.stop_word_list = ['a', 'an', 'the']

    def construct_stop_word_list(self):
        """
        Construct a stop word list including 'a', 'an', 'the'.
        :return: a list of stop words
        """
        return self.stop_word_list

    def remove_stop_words(self, string_list, stop_word_list):
        """
        Remove all the stop words from the list of strings.
        :param string_list: a list of strings
        :param stop_word_list: a list of stop words
        :return: a list of words without stop words
        """
        result = []
        for string in string_list:
            for word in re.split(r'\W+', string):
                if word.lower() not in stop_word_list:
                    result.append(word)
        return result

    def process(self, string_list):
        """
        Construct a stop word list including 'a', 'an', 'the', and remove all the stop words from the list of strings.
        :param string_list: a list of strings
        :return: a list of words without stop words
        """
        stop_word_list = self.construct_stop_word_list()
        return self.remove_stop_words(string_list, stop_word_list)
    
if __name__ == "__main__":
    processor = NLPDataProcessor()
    sample_strings = ["This is a test.", "An example of stop words."]
    processed_words = processor.process(sample_strings)
    print(processed_words)  # Output: ['This', 'is', 'test', 'example', 'of', 'stop', 'words']