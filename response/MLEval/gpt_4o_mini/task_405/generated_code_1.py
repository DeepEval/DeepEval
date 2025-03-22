from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def method():
    # Load dataset (for the sake of example, we'll use the iris dataset)
    data = load_iris()
    X = data.data  # Features
    y = data.target  # Target labels

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Return the trained model for validation purposes
    return model

# Call the method for validation
output = method()
print("Model trained successfully:", output)