import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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
        # Download the stop words list if not already downloaded
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        stop_words.update(['a', 'an', 'the'])
        return list(stop_words)

    def remove_stop_words(self, string_list, stop_word_list):
        """
        Remove all the stop words from the list of strings.
        :param string_list: a list of strings
        :param stop_word_list: a list of stop words
        :return: a list of words without stop words
        >>> NLPDataProcessor.remove_stop_words(['This is a test.'], ['a', 'an', 'the'])
        [['This', 'is', 'test.']]
        """
        # Tokenize the strings
        tokenized_strings = [word_tokenize(string) for string in string_list]
        
        # Remove stop words
        filtered_strings = [[word for word in string if word not in stop_word_list] for string in tokenized_strings]
        
        return filtered_strings

    def process(self, string_list):
        """
        Construct a stop word list including 'a', 'an', 'the', and remove all the stop words from the list of strings.
        :param string_list: a list of strings
        :return: a list of words without stop words
        >>> NLPDataProcessor.process(['This is a test.'])
        [['This', 'is', 'test.']]
        """
        stop_word_list = self.construct_stop_word_list()
        return self.remove_stop_words(string_list, stop_word_list)


if __name__ == "__main__":
    instance = NLPDataProcessor()
    
    # Test case for construct_stop_word_list method
    output = instance.construct_stop_word_list()
    print("construct_stop_word_list():", output)
    
    # Test case for remove_stop_words method
    string_list = ['This is a test.']
    output = instance.remove_stop_words(string_list, instance.construct_stop_word_list())
    print("remove_stop_words():", output)
    
    # Test case for process method
    string_list = ['This is a test.']
    output = instance.process(string_list)
    print("process():", output)