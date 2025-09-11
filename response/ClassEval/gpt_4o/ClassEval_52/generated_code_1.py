import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string

# Ensure necessary NLTK data packages are downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

class Lemmatization:
    """
    This is a class about Lemmatization, which utilizes the nltk library to perform lemmatization and part-of-speech tagging on sentences, as well as remove punctuation.
    """

    def __init__(self):
        """
        Creates a WordNetLemmatizer object and stores it in the self.lemmatizer member variable.
        """
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_sentence(self, sentence):
        """
        Remove punctuations of the sentence and tokenizes the input sentence, mark the part of speech tag of each word,
        lemmatizes the words with different parameters based on their parts of speech, and stores them in a list.
        :param sentence: a sentence str
        :return: a list of words which have been lemmatized.
        """
        sentence = self.remove_punctuation(sentence)
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)

        lemmatized_words = []
        for word, tag in pos_tags:
            pos = self.get_wordnet_pos(tag)
            lemmatized_words.append(self.lemmatizer.lemmatize(word, pos))
        
        return lemmatized_words

    def get_pos_tag(self, sentence):
        """
        Remove punctuations of the sentence and tokenizes the input sentence, mark the part of speech tag of each word.
        :param sentence: a sentence str
        :return: list, part of speech tag of each word in the sentence.
        """
        sentence = self.remove_punctuation(sentence)
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        
        return [tag for _, tag in pos_tags]

    def remove_punctuation(self, sentence):
        """
        Removes punctuation from the input text.
        :param sentence: a sentence str
        :return: str, sentence without any punctuation
        """
        return sentence.translate(str.maketrans('', '', string.punctuation))

    def get_wordnet_pos(self, treebank_tag):
        """
        Convert treebank tags to WordNet part of speech names.
        :param treebank_tag: str, treebank tag
        :return: WordNet tag
        """
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN

if __name__ == "__main__":
    lemmatization = Lemmatization()

    # Test case for lemmatize_sentence
    sentence = "I am running in a race."
    output = lemmatization.lemmatize_sentence(sentence)
    print(output)  # Output: ['I', 'be', 'run', 'in', 'a', 'race']

    # Test case for get_pos_tag
    sentence = "I am running in a race."
    output = lemmatization.get_pos_tag(sentence)
    print(output)  # Output: ['PRP', 'VBP', 'VBG', 'IN', 'DT', 'NN']

    # Test case for remove_punctuation
    sentence = "I am running in a race."
    output = lemmatization.remove_punctuation(sentence)
    print(output)  # Output: 'I am running in a race'