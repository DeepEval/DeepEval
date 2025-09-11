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
        # Remove punctuation and tokenize the sentence
        words = word_tokenize(sentence.lower())
        # Tag parts of speech
        pos_tags = pos_tag(words)
        # Lemmatize words, adjusting based on POS
        lemmatized_words = []
        for word, pos in pos_tags:
            lemmatized_word = self.lemmatizer.lemmatize(word, pos=pos)
            lemmatized_words.append(lemmatized_word)
        return lemmatized_words

    def get_pos_tag(self, sentence):
        # Remove punctuation and tokenize the sentence
        words = word_tokenize(sentence.lower())
        # Tag parts of speech
        pos_tags = pos_tag(words)
        return pos_tags

    def remove_punctuation(self, sentence):
        # Remove punctuation
        return ''.join(ch for ch in sentence if ch not in string.punctuation)

# Test cases
if __name__ == "__main__":
    lemmatization = Lemmatization()
    print(lemmatization.lemmatize_sentence("I am running in a race."))  # ['i', 'be', 'run', 'in', 'a', 'race']
    pos_tags = lemmatization.get_pos_tag("I am running in a race.")  # ['PRP', 'VBP', 'VBG', 'IN', 'DT', 'NN']
    print(lemmatization.remove_punctuation("I am running in a race."))  # 'I am running in a race'