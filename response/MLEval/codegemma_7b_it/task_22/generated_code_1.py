import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import collections
from wordcloud import WordCloud

def method():

    # Read the question pairs
    questions = []
    # with open('question_pairs.txt', 'r') as f:
    # 修改为本地数据文件
    with open('evaluation/dynamic_checking/baselines/MLEval/codegemma_7b_it/testcases/task_22.txt', 'r') as f:
        for line in f:
            question = line.strip().split('\t')
            questions.append(question)

    # Extract the text from the question pairs
    text = []
    for question in questions:
        text.append(question[0])
        text.append(question[1])

    # Tokenize the text
    tokens = word_tokenize(' '.join(text))

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stem the tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Create a word cloud for duplicates
    duplicates = [question[0] for question in questions if question[1] == question[0]]
    duplicates_tokens = word_tokenize(' '.join(duplicates))
    duplicates_tokens = [token for token in duplicates_tokens if token not in stop_words]
    duplicates_tokens = [stemmer.stem(token) for token in duplicates_tokens]
    duplicates_wordcloud = WordCloud(background_color='white', max_words=50, stopwords=stop_words).generate(' '.join(duplicates_tokens))

    # Create a word cloud for non-duplicates
    non_duplicates = [question[0] for question in questions if question[1] != question[0]]
    non_duplicates_tokens = word_tokenize(' '.join(non_duplicates))
    non_duplicates_tokens = [token for token in non_duplicates_tokens if token not in stop_words]
    non_duplicates_tokens = [stemmer.stem(token) for token in non_duplicates_tokens]
    non_duplicates_wordcloud = WordCloud(background_color='white', max_words=50, stopwords=stop_words).generate(' '.join(non_duplicates_tokens))

    # Display the word clouds
    print('Word Cloud of Duplicates:')
    print(duplicates_wordcloud)

    print('\nWord Cloud of Non-Duplicates:')
    print(non_duplicates_wordcloud)

method()