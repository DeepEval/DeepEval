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
        """
        # Remove punctuation
        sentence = self.remove_punctuation(sentence)
        # Tokenize the sentence
        words = word_tokenize(sentence)
        # Get the part of speech tag for each word
        pos_tags = pos_tag(words)
        # Lemmatize the words based on their part of speech
        lemmatized_words = []
        for word, pos in pos_tags:
            if pos.startswith('N'):  # Noun
                lemmatized_word = self.lemmatizer.lemmatize(word, 'n')
            elif pos.startswith('V'):  # Verb
                lemmatized_word = self.lemmatizer.lemmatize(word, 'v')
            elif pos.startswith('J'):  # Adjective
                lemmatized_word = self.lemmatizer.lemmatize(word, 'a')
            elif pos.startswith('R'):  # Adverb
                lemmatized_word = self.lemmatizer.lemmatize(word, 'r')
            else:
                lemmatized_word = self.lemmatizer.lemmatize(word)
            lemmatized_words.append(lemmatized_word)
        return lemmatized_words

    def get_pos_tag(self, sentence):
        """
        Remove punctuations of the sentence and tokenizes the input sentence, mark the part of speech tag of each word.
        :param sentence: a sentence str
        :return: list, part of speech tag of each word in the sentence.
        """
        # Remove punctuation
        sentence = self.remove_punctuation(sentence)
        # Tokenize the sentence
        words = word_tokenize(sentence)
        # Get the part of speech tag for each word
        pos_tags = pos_tag(words)
        return [pos[1] for pos in pos_tags]

    def remove_punctuation(self, sentence):
        """
        Removes punctuation from the input text.
        :param sentence: a sentence str
        :return: str, sentence without any punctuation
        """
        return sentence.translate(str.maketrans('', '', string.punctuation))

if __name__ == "__main__":
    lemmatization = Lemmatization()
    print(lemmatization.lemmatize_sentence("I am running in a race."))
    print(lemmatization.get_pos_tag("I am running in a race."))
    print(lemmatization.remove_punctuation("I am running in a race."))