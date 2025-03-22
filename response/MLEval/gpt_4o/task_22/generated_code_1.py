import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def method():
    # Simulating a DataFrame with question pairs and a duplicate flag
    data = {
        'question1': [
            "What is the capital of France?",
            "How to learn Python?",
            "What is the capital of France?",
            "How to learn Python?",
            "What is AI?"
        ],
        'question2': [
            "Where is the capital of France located?",
            "Steps to learn Python programming",
            "What is the capital city of France?",
            "Best way to learn Python?",
            "Define artificial intelligence"
        ],
        'is_duplicate': [1, 1, 1, 0, 0]
    }

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Separate the data into duplicates and non-duplicates
    duplicates = df[df['is_duplicate'] == 1]
    non_duplicates = df[df['is_duplicate'] == 0]

    # Combine questions for word cloud generation
    duplicate_text = ' '.join(duplicates['question1'].tolist() + duplicates['question2'].tolist())
    non_duplicate_text = ' '.join(non_duplicates['question1'].tolist() + non_duplicates['question2'].tolist())

    # Generate word clouds
    duplicate_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(duplicate_text)
    non_duplicate_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_duplicate_text)

    # Plot the word clouds
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(duplicate_wordcloud, interpolation='bilinear')
    plt.title("Duplicates Word Cloud")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(non_duplicate_wordcloud, interpolation='bilinear')
    plt.title("Non-Duplicates Word Cloud")
    plt.axis('off')

    plt.show()

    # Output can be returned if needed, though not necessary for visualizations
    output = {
        "duplicates_wordcloud": duplicate_wordcloud,
        "non_duplicates_wordcloud": non_duplicate_wordcloud
    }

    return output

# Call the method for validation
output = method()