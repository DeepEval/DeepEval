import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

def method():
    # Sample data for demonstration purposes
    data = {
        'question1': ['What is the capital of France?', 'How do you make a cake?', 'What is machine learning?'],
        'question2': ['What is the capital of France?', 'How do you make a cake step by step?', 'What are deep learning?'],
        'is_duplicate': [0, 1, 0]
    }
    df = pd.DataFrame(data)

    # Combine all questions into a single text
    all_questions = df['question1'] + ' ' + df['question2']

    # Separate into duplicate and non-duplicate questions
    duplicate_questions = df[df['is_duplicate'] == 1]['question1'] + ' ' + df[df['is_duplicate'] == 1]['question2']
    non_duplicate_questions = df[df['is_duplicate'] == 0]['question1'] + ' ' + df[df['is_duplicate'] == 0]['question2']

    # Generate word clouds
    wordcloud_duplicate = WordCloud(width=800, height=400, background_color='white').generate(' '.join(duplicate_questions))
    wordcloud_non_duplicate = WordCloud(width=800, height=400, background_color='white').generate(' '.join(non_duplicate_questions))

    # Plot the word clouds
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.title('Word Cloud for Duplicate Questions')
    plt.imshow(wordcloud_duplicate, interpolation='bilinear')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Word Cloud for Non-Duplicate Questions')
    plt.imshow(wordcloud_non_duplicate, interpolation='bilinear')
    plt.axis('off')

    plt.show()

    # Return the word clouds (optional, as they are displayed directly)
    return {
        'duplicate_wordcloud': wordcloud_duplicate,
        'non_duplicate_wordcloud': wordcloud_non_duplicate
    }

# Call the method for validation
output = method()