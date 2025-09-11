import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

class Lemmatization:
    """
    This is a class about Lemmatization, which utilizes the nltk library to perform lemmatization and part-of-speech tagging on sentences, as well as remove punctuation.
    """

    def __init__(self):
        """
        creates a WordNetLemmatizer object and stores it in the self.lemmatizer member variable.
        """
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_sentence(self, sentence):
        """
        Remove punctuations of the sentence and tokenizes the input sentence, mark the part of speech tag of each word,
        lemmatizes the words with different parameters based on their parts of speech, and stores in a list.
        :param sentence: a sentence str
        :return: a list of words which have been lemmatized.
        >>> lemmatization = Lemmatization()
        >>> lemmatization.lemmatize_sentence("I am running in a race.")
        ['I', 'be', 'run', 'in', 'a', 'race']
        """
        # Remove punctuation
        sentence_no_punct = self.remove_punctuation(sentence)
        # Tokenize the sentence
        tokens = word_tokenize(sentence_no_punct)
        # Get part of speech tags
        pos_tags = self.get_pos_tag(sentence_no_punct)
        # Lemmatize the words based on their part of speech
        lemmatized_words = []
        for word, tag in zip(tokens, pos_tags):
            if tag.startswith('VB'):
                lemmatized_word = self.lemmatizer.lemmatize(word, pos='v')
            else:
                lemmatized_word = self.lemmatizer.lemmatize(word)
            lemmatized_words.append(lemmatized_word)
        return lemmatized_words

    def get_pos_tag(self, sentence):
        """
        Remove punctuations of the sentence and tokenizes the input sentence, mark the part of speech tag of each word.
        :param sentence: a sentence str
        :return: list, part of speech tag of each word in the sentence.
        >>> lemmatization = Lemmatization()
        >>> lemmatization.get_pos_tag("I am running in a race.")
        ['PRP', 'VBP', 'VBG', 'IN', 'DT', 'NN']
        """
        # Remove punctuation
        sentence_no_punct = self.remove_punctuation(sentence)
        # Tokenize the sentence
        tokens = word_tokenize(sentence_no_punct)
        # Get part of speech tags
        pos_tags = pos_tag(tokens)
        return [tag for _, tag in pos_tags]

    def remove_punctuation(self, sentence):
        """
        Removes punctuation from the input text.
        :param sentence: a sentence str
        :return: str, sentence without any punctuation
        >>> lemmatization = Lemmatization()
        >>> lemmatization.remove_punctuation("I am running in a race.")
        'I am running in a race'
        """
        return sentence.translate(str.maketrans('', '', string.punctuation))


if __name__ == "__main__":
    lemmatization = Lemmatization()

    # Test case for remove_punctuation
    output_remove_punct = lemmatization.remove_punctuation("I am running in a race.")
    print("Remove Punctuation Output:", output_remove_punct)

    # Test case for get_pos_tag
    output_pos_tag = lemmatization.get_pos_tag("I am running in a race.")
    print("POS Tag Output:", output_pos_tag)

    # Test case for lemmatize_sentence
    output_lemmatize_sentence = lemmatization.lemmatize_sentence("I am running in a race.")
    print("Lemmatize Sentence Output:", output_lemmatize_sentence)