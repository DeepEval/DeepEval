import re

class SplitSentence:
    def split_sentences(self, sentences_string):
        """
        Split a string into a list of sentences. Sentences end with . or ? and
        with a space after that.
        :param sentences_string: string, string to split
        :return: list, split sentence list
        """
        sentences = re.split(r'[.?!]\s+', sentences_string)
        return [s.strip() for s in sentences]

    def count_words(self, sentence):
        """
        Count the number of words in a sentence.
        :param sentence: string, sentence to be counted, where words are separated
        by spaces
        :return: int, number of words in the sentence
        """
        words = re.split(r'\s+', sentence)
        return len(words)

    def process_text_file(self, sentences_string):
        """
        Given a text, return the number of words in the longest sentence.
        :param sentences_string: string, undivided long sentence
        :return: int, the number of words in the longest sentence
        """
        sentences = self.split_sentences(sentences_string)
        max_length = 0
        for sentence in sentences:
            length = self.count_words(sentence)
            if length > max_length:
                max_length = length
        return max_length

if __name__ == "__main__":
    ss = SplitSentence()
    sentences_string = "aaa aaaa. bb bbbb bbb? cccc ccccccc cc ccc. dd ddd?"
    print(ss.split_sentences(sentences_string))
    print(ss.count_words("abc def"))
    print(ss.process_text_file(sentences_string))