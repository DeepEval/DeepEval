from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming you have your data in X and y variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def method():
    # Create an instance of the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)  
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    return model

# Call the method and store the trained model
trained_model = method()

# Make predictions on the test data
y_pred = trained_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)