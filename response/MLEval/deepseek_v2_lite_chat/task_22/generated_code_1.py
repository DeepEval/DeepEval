import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def method():
    # Data preparation
    # Replace this with your actual data
    questions = [
        "What is the capital of France?",
        "Who is the current president of France?",
        "What is the capital of France?",
        "Who is the current president of France?",
        "Who is the current president of France?"
    ]
    
    # Convert lists to strings
    questions_str = [" ".join(questions)]
    
    # Vectorize the text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(questions_str)
    word_counts = X.toarray()
    
    # Count the frequency of words
    word_counts_flat = np.hstack(word_counts).flatten()
    word_counts_dict = dict(zip(vectorizer.get_feature_names_out(), word_counts_flat))
    
    # Create a list of tuples (word, count)
    word_count_pairs = [(word, count) for word, count in word_counts_dict.items()]
    
    # Sort the pairs by count in descending order
    word_count_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Create the word cloud
    wordcloud = WordCloud(width=800, height=400, max_words=20, background_color="white").generate_from_frequencies(word_count_pairs)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    # Filter out non-duplicates and duplicates
    non_duplicates = [pair for pair in word_count_pairs if questions.count(questions[0]) == 1]
    duplicates = [pair for pair in word_count_pairs if questions.count(questions[0]) > 1]
    
    # Output the results
    output = {
        'non_duplicates': non_duplicates,
        'duplicates': duplicates
    }
    
    return output

# Call the method for validation
validation_result = method()
print(validation_result)