import re
from collections import Counter
from typing import List, Dict

class NLPDataProcessor2:
    """
    The class processes NLP data by extracting words from a list of strings, calculating the frequency of each word, and returning the top 5 most frequent words.
    """

    def process_data(self, string_list: List[str]) -> List[List[str]]:
        """
        Keep only English letters and spaces in the string, then convert the string to lower case, and then split the string into a list of words.
        
        :param string_list: a list of strings
        :return: words_list: a list of words lists
        >>> NLPDataProcessor2().process_data(['This is a test.'])
        [['this', 'is', 'a', 'test']]
        """

        words_list = []
        for string in string_list:
            # Remove non-English letters and convert to lower case
            cleaned_string = re.sub('[^a-zA-Z\s]', '', string).lower()
            # Split the string into a list of words
            words = cleaned_string.split()
            words_list.append(words)
        return words_list

    def calculate_word_frequency(self, words_list: List[List[str]]) -> Dict[str, int]:
        """
        Calculate the word frequency of each word in the list of words list, and sort the word frequency dictionary by value in descending order.
        
        :param words_list: a list of words lists
        :return: top 5 word frequency dictionary, a dictionary of word frequency, key is word, value is frequency
        >>> NLPDataProcessor2().calculate_word_frequency([['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']])
        {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}
        """

        # Flatten the list of words lists into a single list of words
        flat_words = [word for sublist in words_list for word in sublist]
        
        # Calculate the word frequency using Counter
        word_frequency = Counter(flat_words)
        
        # Sort the word frequency dictionary by value in descending order
        sorted_word_frequency = dict(sorted(word_frequency.items(), key=lambda item: item[1], reverse=True))
        
        # Return the top 5 most frequent words
        return dict(list(sorted_word_frequency.items())[:5])

    def process(self, string_list: List[str]) -> Dict[str, int]:
        """
        Keep only English letters and spaces in the string, then convert the string to lower case, and then split the string into a list of words. 
        Calculate the word frequency of each word in the list of words list, and sort the word frequency dictionary by value in descending order.
        
        :param string_list: a list of strings
        :return: top 5 word frequency dictionary, a dictionary of word frequency, key is word, value is frequency
        >>> NLPDataProcessor2().process(['This is a test.', 'This is another test.'])
        {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}
        """

        words_list = self.process_data(string_list)
        return self.calculate_word_frequency(words_list)

if __name__ == "__main__":
    instance = NLPDataProcessor2()
    
    # Test case for process_data
    string_list = ['This is a test.']
    output = instance.process_data(string_list)
    print(output)  # Output: [['this', 'is', 'a', 'test']]
    
    # Test case for calculate_word_frequency
    words_list = [['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']]
    output = instance.calculate_word_frequency(words_list)
    print(output)  # Output: {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}
    
    # Test case for process
    string_list = ['This is a test.', 'This is another test.']
    output = instance.process(string_list)
    print(output)  # Output: {'this': 2, 'is': 2, 'test': 2, 'a': 1, 'another': 1}