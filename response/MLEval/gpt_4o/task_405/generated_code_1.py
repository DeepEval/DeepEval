# Import necessary packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def method():
    # Load sample data
    data = load_iris()
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)

    # Fit the model to the training data
    clf.fit(X_train, y_train)

    # Return the trained model
    return clf

# Call the method for validation
trained_model = method()
print(f"Trained RandomForestClassifier: {trained_model}")