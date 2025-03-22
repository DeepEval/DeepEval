# Import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import cross_val_score

# Function to create an Extra Tree model
def method():
    """
    This function creates an Extra Tree model, trains it on the Iris dataset,
    and returns the accuracy of the model.
    
    Returns:
        output (float): The accuracy of the Extra Tree model.
    """

    # Load the Iris dataset
    iris = load_iris()
    
    # Convert the dataset to a DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Split the dataset into features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create an Extra Trees Classifier model
    model = ExtraTreesClassifier(n_estimators=10, random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print the accuracy of the model
    print(f"Accuracy: {accuracy:.2f}")
    
    # Use cross-validation to evaluate the model's performance
    scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Average cross-validation score: {scores.mean():.2f}")
    
    # Return the accuracy of the model
    return accuracy

# Call the method for validation
method()