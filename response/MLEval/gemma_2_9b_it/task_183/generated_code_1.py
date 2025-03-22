import pandas as pd
from sklearn.tree import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():
    # Load your dataset here. 
    # Replace 'your_dataset.csv' with the actual path to your dataset.
    data = pd.read_csv('your_dataset.csv') 

    # Separate features (X) and target variable (y)
    X = data.drop('target_variable', axis=1)  # Replace 'target_variable' with your target column name
    y = data['target_variable']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an Extra Trees Classifier
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)  # Adjust hyperparameters as needed

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Return the accuracy score (optional)
    return accuracy

# Call the method to run the code
output = method()