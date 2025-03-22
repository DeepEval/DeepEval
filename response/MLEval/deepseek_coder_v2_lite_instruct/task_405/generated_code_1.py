from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def method():
    # Load the iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create an instance of RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model to the training data
    clf.fit(X_train, y_train)

    # Predict the test set
    y_pred = clf.predict(X_test)

    # For validation purposes, let's print the first 10 predictions
    print(y_pred[:10])

    # Return the output if needed
    output = clf
    return output

# Call the method for validation
method()