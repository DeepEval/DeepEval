import re

class SplitSentence:
    """
    The class allows to split sentences, count words in a sentence, and process a text file to find the maximum word count.
    """


    def split_sentences(self, sentences_string):
        """
        Split a string into a list of sentences. Sentences end with . or ? and with a space after that. Please note that Mr. also end with . but are not sentences.
        :param sentences_string: string, string to split
        :return:list, split sentence list
        >>> ss = SplitSentence()
        >>> ss.split_sentences("aaa aaaa. bb bbbb bbb? cccc cccc. dd ddd?")
        ['aaa aaaa.', 'bb bbbb bbb?', 'cccc cccc.', 'dd ddd?']
        """
        return re.findall(r'[^.?!]+[.?!]\s', sentences_string)

    def count_words(self, sentence):
        """
        Count the number of words in a sentence. Note that words are separated by spaces and that punctuation marks and numbers are not counted as words.
        :param sentence:string, sentence to be counted, where words are separated by spaces
        :return:int, number of words in the sentence
        >>> ss.count_words("abc def")
        2
        """
        return len(sentence.split())

    def process_text_file(self, sentences_string):
        """
        Given a text, return the number of words in the longest sentence
        :param sentences_string: string, undivided long sentence
        :return:int, the number of words in the longest sentence
        >>> ss.process_text_file("aaa aaaa. bb bbbb bbb? cccc ccccccc cc ccc. dd ddd?")
        4
        """
        sentences = self.split_sentences(sentences_string)
        max_words = 0
        for sentence in sentences:
            words = self.count_words(sentence)
            if words > max_words:
                max_words = words
        return max_words
    
if __name__ == "__main__":
    instance = SplitSentence()
    # Test case for split_sentences
    output = instance.split_sentences("aaa aaaa. bb bbbb bbb? cccc cccc. dd ddd?")
    print(f"Split Sentences: {output}")

    # Test case for count_words
    output = instance.count_words("abc def")
    print(f"Word Count: {output}")

    # Test case for process_text_file
    output = instance.process_text_file("aaa aaaa. bb bbbb bbb? cccc ccccccc cc ccc. dd ddd?")
    print(f"Max Word Count: {output}")