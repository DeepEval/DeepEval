import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def method():
    # Sample data: Replace this with your actual data
    data = {
        'question': [
            "What is the capital of France?",
            "What is the capital of Germany?",
            "What is the capital of France?",  # Duplicate
            "What is the capital of Italy?",
            "How to learn Python?",
            "How to learn Python?",  # Duplicate
            "What is the capital of Germany?",  # Duplicate
            "What is machine learning?",
            "What is machine learning?"  # Duplicate
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Identify duplicates
    duplicates = df[df.duplicated('question', keep=False)]
    non_duplicates = df[~df.duplicated('question', keep=False)]
    
    # Create text for word clouds
    duplicates_text = ' '.join(duplicates['question'].tolist())
    non_duplicates_text = ' '.join(non_duplicates['question'].tolist())
    
    # Generate word clouds
    generate_wordcloud(duplicates_text, 'Word Cloud of Duplicate Question Pairs')
    generate_wordcloud(non_duplicates_text, 'Word Cloud of Non-Duplicate Question Pairs')
    
    # Return output if needed (e.g., could be the dataframes or any specific result)
    output = {
        "duplicates": duplicates,
        "non_duplicates": non_duplicates
    }
    
    return output

# Call the method for validation
output = method()