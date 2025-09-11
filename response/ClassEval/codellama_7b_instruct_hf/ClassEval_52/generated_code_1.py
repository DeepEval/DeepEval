import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string

class Lemmatization:
    """
    This is a class about Lemmatization, which utilizes the nltk library to perform
    lemmatization and part-of-speech tagging on sentences, as well as remove punctuation.
    """

    def __init__(self):
        """
        creates a WordNetLemmatizer object and stores it in the self.lemmatizer
        member variable.
        """
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_sentence(self, sentence):
        """
        Remove punctuations of the sentence and tokenizes the input sentence, mark
        the part of speech tag of each word,
        lemmatizes the words with different parameters based on their parts of
        speech, and stores in a list.
        :param sentence: a sentence str
        :return: a list of words which have been lemmatized.
        >>> lemmatization = Lemmatization()
        >>> lemmatization.lemmatize_sentence("I am running in a race.")
        ['I', 'be', 'run', 'in', 'a', 'race']

        """
        sentence = sentence.lower()
        sentence = self.remove_punctuation(sentence)
        tokenized_sentence = word_tokenize(sentence)
        pos_tagged_sentence = pos_tag(tokenized_sentence)
        lemmatized_words = []
        for word, tag in pos_tagged_sentence:
            if tag.startswith('NN'):
                lemmatized_words.append(self.lemmatizer.lemmatize(word, 'n'))
            elif tag.startswith('VB'):
                lemmatized_words.append(self.lemmatizer.lemmatize(word, 'v'))
            elif tag.startswith('JJ'):
                lemmatized_words.append(self.lemmatizer.lemmatize(word, 'a'))
            else:
                lemmatized_words.append(word)
        return lemmatized_words

    def get_pos_tag(self, sentence):
        """
        Remove punctuations of the sentence and tokenizes the input sentence, mark
        the part of speech tag of each word.
        :param sentence: a sentence str
        :return: list, part of speech tag of each word in the sentence.
        >>> lemmatization = Lemmatization()
        >>> lemmatization.get_pos_tag("I am running in a race.")
        ['PRP', 'VBP', 'VBG', 'IN', 'DT', 'NN']

        """
        sentence = sentence.lower()
        sentence = self.remove_punctuation(sentence)
        tokenized_sentence = word_tokenize(sentence)
        pos_tagged_sentence = pos_tag(tokenized_sentence)
        return [tag for word, tag in pos_tagged_sentence]

    def remove_punctuation(self, sentence):
        """
        Removes punctuation from the input text.
        :param sentence: a sentence str
        :return: str, sentence without any punctuation
        >>> lemmatization = Lemmatization()
        >>> lemmatization.remove_punctuation("I am running in a race.")
        'I am running in a race'

        """
        translator = str.maketrans('', '', string.punctuation)
        return sentence.translate(translator)

if __name__ == "__main__":
    lemmatization = Lemmatization()
    output = lemmatization.lemmatize_sentence("I am running in a race.")
    print(output)
    output = lemmatization.get_pos_tag("I am running in a race.")
    print(output)
    output = lemmatization.remove_punctuation("I am running in a race.")
    print(output)