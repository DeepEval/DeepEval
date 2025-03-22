import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def method():
    # Sample data (replace this with your actual training data)
    data = pd.DataFrame({
        'text': ['This is a positive sentence.', 'This is a negative sentence.', 'Another positive example.', 'This is neutral.'],
        'label': [1, 0, 1, 2]  # Assuming 1 for positive, 0 for negative, 2 for neutral
    })

    # Create a pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  
        ('clf', LogisticRegression())
    ])

    # Fit the pipeline to the training data
    pipeline.fit(data['text'], data['label'])

    return pipeline

# Call the method and validate
trained_pipeline = method() 

# Example validation (replace with your actual validation data)
validation_data = pd.DataFrame({
    'text': ['This is a great movie!', 'I disliked this product.', 'Neutral opinion here.']
})

# Make predictions
predictions = trained_pipeline.predict(validation_data['text'])

print(predictions)