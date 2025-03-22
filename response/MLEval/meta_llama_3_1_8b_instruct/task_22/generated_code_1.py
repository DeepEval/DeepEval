# Import necessary packages
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter
import numpy as np

# Sample data (you should replace this with your actual data)
question_pairs = [
    "What is machine learning? What is machine learning?",
    "What is Python? What is Python?",
    "What is AI? What is AI?",
    "What is NLP? What is NLP?",
    "What is ML? What is ML?",
    "What is DL? What is DL?",
    "What is computer science? What is computer science?",
    "What is data science? What is data science?",
]

def method():
    # Remove punctuation and special characters
    text =''.join(question_pairs).lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split the text into words
    words = text.split()
    
    # Count the frequency of each word
    word_counts = Counter(words)
    
    # Generate word cloud for duplicates
    duplicate_words =''.join([word * 2 for word in word_counts.keys() if word_counts[word] > 1])
    duplicate_wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(duplicate_words)
    plt.figure(figsize=(10,5))
    plt.imshow(duplicate_wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title("Word Cloud of Duplicates")
    plt.show()
    
    # Generate word cloud for non-duplicates
    non_duplicate_words =''.join([word for word in word_counts.keys() if word_counts[word] == 1])
    non_duplicate_wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(non_duplicate_words)
    plt.figure(figsize=(10,5))
    plt.imshow(non_duplicate_wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title("Word Cloud of Non-Duplicates")
    plt.show()
    
    # Find the most frequent occurring words
    max_count = max(word_counts.values())
    most_frequent_words = {word: count for word, count in word_counts.items() if count == max_count}
    
    return most_frequent_words

# Call the generated method for validation
output = method()
print(output)