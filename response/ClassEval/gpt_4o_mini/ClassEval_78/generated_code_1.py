import re

class SplitSentence:
    """
    The class allows to split sentences, count words in a sentence, and process a text file to find the maximum word count.
    """

    def split_sentences(self, sentences_string):
        """
        Split a string into a list of sentences. Sentences end with . or ? and with a space after that. 
        Please note that Mr. also ends with . but are not sentences.
        :param sentences_string: string, string to split
        :return: list, split sentence list
        >>> ss = SplitSentence()
        >>> ss.split_sentences("aaa aaaa. bb bbbb bbb? cccc cccc. dd ddd?")
        ['aaa aaaa.', 'bb bbbb bbb?', 'cccc cccc.', 'dd ddd?']
        """
        # Regex to split sentences that end with '.' or '?' followed by a space or end of string.
        sentences = re.split(r'(?<!Mr)\. |[?]', sentences_string.strip())
        return [s.strip() for s in sentences if s]

    def count_words(self, sentence):
        """
        Count the number of words in a sentence. Note that words are separated by spaces and that punctuation marks and numbers are not counted as words.
        :param sentence: string, sentence to be counted, where words are separated by spaces
        :return: int, number of words in the sentence
        >>> ss = SplitSentence()
        >>> ss.count_words("abc def")
        2
        """
        # Using regex to count words, which are sequences of alphabetic characters.
        words = re.findall(r'\b[a-zA-Z]+\b', sentence)
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
        max_word_count = 0
        for sentence in sentences:
            word_count = self.count_words(sentence)
            max_word_count = max(max_word_count, word_count)
        return max_word_count


if __name__ == "__main__":
    ss = SplitSentence()
    
    # Test case for split_sentences
    output_split = ss.split_sentences("aaa aaaa. bb bbbb bbb? cccc cccc. dd ddd?")
    print("Split Sentences Output:", output_split)
    
    # Test case for count_words
    output_count = ss.count_words("abc def")
    print("Count Words Output:", output_count)
    
    # Test case for process_text_file
    output_process = ss.process_text_file("aaa aaaa. bb bbbb bbb? cccc ccccccc cc ccc. dd ddd?")
    print("Process Text File Output:", output_process)