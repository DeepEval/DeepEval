from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def method():
    # Sample data (replace with your actual question pairs)
    duplicates = [
        ("What is the capital of France?", "What is the capital of France?"),
        ("How many continents are there?", "How many continents are there?"),
        ("Who painted the Mona Lisa?", "Who painted the Mona Lisa?")
    ]
    non_duplicates = [
        ("What is the meaning of life?", "How do I bake a cake?"),
        ("What is the weather like today?", "Where is the nearest coffee shop?"),
        ("What is your favorite color?", "What is the tallest mountain?")
    ]

    # Combine and preprocess text
    all_text = " ".join([question + " " + answer for question, answer in duplicates + non_duplicates])
    
    # Create WordCloud object
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white', 
                          stopwords=STOPWORDS,
                          min_font_size=10).generate(all_text)

    # Display the generated image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    # You can also save the word cloud as an image
    # wordcloud.to_file("wordcloud.png")

    return "Word cloud generated successfully!"

# Call the method to generate the word cloud
output = method()
print(output)