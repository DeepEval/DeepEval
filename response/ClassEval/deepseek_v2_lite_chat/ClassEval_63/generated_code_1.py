import re
from collections import Counter

class NLPDataProcessor2:
    def process_data(self, string_list):
        """
        Extracts words from a list of strings, converts them to lower case, and returns a list of words.
        :param string_list: A list of strings
        :return: A list of words lists
        """
        words_list = []
        for string in string_list:
            # Keep only English letters and spaces
            cleaned_string = re.sub(r'[^\w\s]', '', string)
            # Convert to lower case
            cleaned_string = cleaned_string.lower()
            # Split the string into words
            words_list.append(cleaned_string.split())
        return words_list

    def calculate_word_frequency(self, words_list):
        """
        Calculates the frequency of each word in the list of words and returns a dictionary sorted by frequency in descending order.
        :param words_list: A list of words lists
        :return: A dictionary of word frequency
        """
        # Flatten the list of lists
        words_flat = [word for words in words_list for word in words]
        # Calculate the frequency of each word
        word_freq = Counter(words_flat)
        # Sort the dictionary by frequency in descending order
        top_5_freq = dict(Counter(words_flat).most_common(5))
        return top_5_freq

    def process(self, string_list):
        """
        Processes data by extracting words, calculating their frequency, and returning the top 5 most frequent words.
        :param string_list: A list of strings
        :return: A dictionary of word frequency
        """
        # Process the data
        words_list = self.process_data(string_list)
        # Calculate the word frequency
        word_freq = self.calculate_word_frequency(words_list)
        return word_freq
    
if __name__ == "__main__":
    processor = NLPDataProcessor2()
    data = ["Hello, world! This is a test.", "This test is only a test."]
    result = processor.process(data)
    print(result)  # Output: {'test': 3, 'this': 2, 'is': 2, 'a': 2, 'hello': 1}