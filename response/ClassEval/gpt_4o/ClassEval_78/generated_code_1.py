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
        # Regular expression to split sentences considering "Mr." not as end of sentence
        sentences = re.split(r'(?<!Mr)(?<!Mr\w)\. |\? ', sentences_string)
        # Add the punctuation back to the end of each split sentence except the last one if it was split
        sentences = [s.strip() + ('.' if sentences_string[sentences_string.find(s) + len(s)] == '.' else '?') for s in sentences if s]
        return sentences

    def count_words(self, sentence):
        """
        Count the number of words in a sentence. Note that words are separated by spaces and that punctuation marks and numbers are not counted as words.
        :param sentence: string, sentence to be counted, where words are separated by spaces
        :return: int, number of words in the sentence
        >>> ss = SplitSentence()
        >>> ss.count_words("abc def")
        2
        """
        # Split the sentence into words using regex to exclude punctuation and numbers
        words = re.findall(r'\b[A-Za-z]+\b', sentence)
        return len(words)

    def process_text_file(self, sentences_string):
        """
        Given a text, return the number of words in the longest sentence
        :param sentences_string: string, undivided long sentence
        :return: int, the number of words in the longest sentence
        >>> ss = SplitSentence()
        >>> ss.process_text_file("aaa aaaa. bb bbbb bbb? cccc ccccccc cc ccc. dd ddd?")
        4
        """
        sentences = self.split_sentences(sentences_string)
        max_word_count = max(self.count_words(sentence) for sentence in sentences)
        return max_word_count

if __name__ == "__main__":
    ss = SplitSentence()

    # Test case for split_sentences
    split_output = ss.split_sentences("aaa aaaa. bb bbbb bbb? cccc cccc. dd ddd?")
    print(split_output)  # Expected: ['aaa aaaa.', 'bb bbbb bbb?', 'cccc cccc.', 'dd ddd?']

    # Test case for count_words
    count_output = ss.count_words("abc def")
    print(count_output)  # Expected: 2

    # Test case for process_text_file
    process_output = ss.process_text_file("aaa aaaa. bb bbbb bbb? cccc ccccccc cc ccc. dd ddd?")
    print(process_output)  # Expected: 4