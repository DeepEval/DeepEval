import re

class SplitSentence:
    """
    The class allows to split sentences, count words in a sentence, and process a text file to find the maximum word count.
    """

    def split_sentences(self, sentences_string):
        """
        Split a string into a list of sentences. Sentences end with . or ? and with a space after that.
        :param sentences_string: string, string to split
        :return: list, split sentence list
        """
        # Regular expression to match sentences
        pattern = r"([.!?])\s|"  # This matches punctuation followed by a space or no character (for direct sentence start)
        sentences = re.split(pattern, sentences_string)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty strings
        return sentences

    def count_words(self, sentence):
        """
        Count the number of words in a sentence. Note that words are separated by spaces and that punctuation marks and numbers are not counted as words.
        :param sentence: string, sentence to be counted, where words are separated by spaces
        :return: int, number of words in the sentence
        """
        return len(sentence.split())

    def process_text_file(self, file_path):
        """
        Given a text, return the number of words in the longest sentence
        :param file_path: string, path to the text file
        :return: int, the number of words in the longest sentence
        """
        with open(file_path, 'r') as file:
            text = file.read()
        sentences = self.split_sentences(text)
        return max(self.count_words(sentence) for sentence in sentences)

# Test cases
if __name__ == "__main__":
    ss = SplitSentence()
    print(ss.split_sentences("aaa aaaa. bb bbbb bbb? cccc cccc. dd ddd?"))  # ['aaa aaaa.', 'bb bbbb bbb?', 'cccc cccc.', 'dd ddd?']
    print(ss.count_words("abc def"))  # 2
    print(ss.process_text_file("input.txt"))  # Longest sentence word count