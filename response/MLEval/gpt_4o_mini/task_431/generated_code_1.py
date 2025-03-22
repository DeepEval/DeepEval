from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

def method():
    # Load dataset (for example purposes, using 20 Newsgroups dataset)
    data = fetch_20newsgroups(subset='train')
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with vectorization and a classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    # Optionally, return the fitted pipeline or its score on the test set
    output = pipeline.score(X_test, y_test)
    return output

# Call the method to validate
result = method()
print("Test set accuracy:", result)