import re

class SplitSentence:
    """
    The class allows to split sentences, count words in a sentence, and process a text file to find the maximum word count.
    """

    def split_sentences(self, sentences_string):
        """
        Split a string into a list of sentences. Sentences end with . or ? and with a space after that. Please note that Mr. also end with . but are not sentences.
        :param sentences_string: string, string to split
        :return: list, split sentence list
        >>> ss = SplitSentence()
        >>> ss.split_sentences("aaa aaaa. bb bbbb bbb? cccc cccc. dd ddd?")
        ['aaa aaaa.', 'bb bbbb bbb?', 'cccc cccc.', 'dd ddd?']
        """
        # Split sentences based on the pattern
        sentences = re.split(r'(?<=[.!?]) +', sentences_string)
        # Filter out sentences that end with Mr.
        filtered_sentences = [sentence for sentence in sentences if not sentence.endswith('Mr.')]
        return filtered_sentences

    def count_words(self, sentence):
        """
        Count the number of words in a sentence. Note that words are separated by spaces and that punctuation marks and numbers are not counted as words.
        :param sentence: string, sentence to be counted, where words are separated by spaces
        :return: int, number of words in the sentence
        >>> ss.count_words("abc def")
        2
        """
        # Split the sentence into words and filter out punctuation
        words = re.findall(r'\b\w+\b', sentence)
        return len(words)

    def process_text_file(self, sentences_string):
        """
        Given a text, return the number of words in the longest sentence
        :param sentences_string: string, undivided long sentence
        :return: int, the number of words in the longest sentence
        >>> ss.process_text_file("aaa aaaa. bb bbbb bbb? cccc ccccccc cc ccc. dd ddd?")
        4
        """
        # Split sentences and filter out punctuation
        sentences = re.split(r'(?<=[.!?]) +', sentences_string)
        # Filter out sentences that end with Mr.
        filtered_sentences = [sentence for sentence in sentences if not sentence.endswith('Mr.')]
        # Count words in each sentence and find the maximum
        max_word_count = 0
        for sentence in filtered_sentences:
            word_count = len(re.findall(r'\b\w+\b', sentence))
            if word_count > max_word_count:
                max_word_count = word_count
        return max_word_count

if __name__ == "__main__":
    ss = SplitSentence()
    # Test cases
    print(ss.split_sentences("aaa aaaa. bb bbbb bbb? cccc cccc. dd ddd?"))  # Expected output: ['aaa aaaa.', 'bb bbbb bbb?', 'cccc cccc.', 'dd ddd?']
    print(ss.count_words("abc def"))  # Expected output: 2
    print(ss.process_text_file("aaa aaaa. bb bbbb bbb? cccc ccccccc cc ccc. dd ddd?"))  # Expected output: 4