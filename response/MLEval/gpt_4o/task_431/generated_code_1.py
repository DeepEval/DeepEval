from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():
    # Sample data
    X = [
        "I love programming",
        "Python is amazing",
        "I hate bugs",
        "Debugging is fun",
        "I enjoy machine learning",
        "This is a bad movie",
        "I love this show"
    ]
    y = [1, 1, 0, 1, 1, 0, 1]  # Sample labels, 1 for positive sentiment, 0 for negative

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create a pipeline that includes vectorization and a Naive Bayes classifier
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    # Validate the model on the test data
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Call the method for validation
output = method()
print(f"Model accuracy: {output}")