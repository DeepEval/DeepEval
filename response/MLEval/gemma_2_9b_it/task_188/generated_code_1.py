from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X_train and Y_train are already defined

def method():
    # Initialize the model
    model = LogisticRegression() 

    # Optimize the model
    model.fit(X_train, Y_train)

    return model

# Call the method and store the trained model
trained_model = method()

# Example usage for validation (assuming you have X_test and Y_test)
y_pred = trained_model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy}")