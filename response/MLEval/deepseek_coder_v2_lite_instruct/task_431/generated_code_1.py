from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def method():
    # Sample data: text documents and their labels
    documents = ["This is a sample document.", "Another example document.", "Sample documents are useful."]
    labels = [0, 1, 0]  # Example labels

    # Split the data into training and testing sets
    train_docs, test_docs, train_labels, test_labels = train_test_split(documents, labels, test_size=0.2, random_state=42)

    # Create a pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression())
    ])

    # Fit the pipeline to the training data
    pipeline.fit(train_docs, train_labels)

    # Predict on the test data
    predictions = pipeline.predict(test_docs)

    # Calculate the accuracy
    accuracy = accuracy_score(test_labels, predictions)

    # Return the final output (accuracy in this case)
    output = accuracy
    return output

# Call the method for validation
output = method()
print(f"Accuracy: {output}")