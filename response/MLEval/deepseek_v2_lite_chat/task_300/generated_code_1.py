from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def method():
    # Load your dataset
    # Assuming 'data' is your dataset loaded as a pandas DataFrame
    # Replace 'data' with your actual dataset
    
    # Example dataset:
    # data = pd.read_csv('your_dataset.csv')
    
    # Features (X) and Target (y)
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with the name of the column you want to predict
    y = data['target_column']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the logistic regression model
    model = LogisticRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Additional outputs if needed
    print("Accuracy of the model:", accuracy)
    
    return accuracy  # If you need to return the accuracy for further processing

# Call the method for validation
output = method()