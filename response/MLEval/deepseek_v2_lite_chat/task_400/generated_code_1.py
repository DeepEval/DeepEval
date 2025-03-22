from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(train_set, train_labels):
    """
    Trains a machine learning model using the provided training data.
    
    Parameters:
    train_set (array-like): The feature data to train on.
    train_labels (array-like): The labels for the training data.
    
    Returns:
    model (object): Trained model.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_set, train_labels, test_size=0.2, random_state=42)
    
    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# Example usage:
if __name__ == "__main__":
    train_set = # Your data here
    train_labels = # Your labels here
    model, accuracy = train_model(train_set, train_labels)
    print(f"Model accuracy: {accuracy:.2f}")