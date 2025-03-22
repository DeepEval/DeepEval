# Import necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Assume that we have a dataset in a pandas DataFrame called 'data' with features 'X' and target 'y'
data = pd.DataFrame({
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'y': [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
})

# Function to train and fit a logistic regression model
def method():
    """
    Train and fit a logistic regression model on the training set.
    
    Returns:
        output (dict): A dictionary containing the trained model, its accuracy on the test set, and the feature importances.
    """
    
    # Split the data into features 'X' and target 'y'
    X = data[['X1', 'X2']]
    y = data['y']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create a logistic regression model
    model = LogisticRegression(max_iter=1000)
    
    # Train the model on the training set
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate the accuracy of the model on the test set
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get the feature importances
    feature_importances = model.coef_[0]
    
    # Return the output as a dictionary
    return {
       'model': model,
        'accuracy': accuracy,
        'feature_importances': feature_importances
    }

# Call the method for validation
output = method()
print(output)