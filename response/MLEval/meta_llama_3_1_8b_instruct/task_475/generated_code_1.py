# Import necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def method():
    # Create a sample dataset for demonstration purposes
    data = {
        'height': [160, 170, 178, 165, 180, 168, 172, 169, 173, 161],
        'weight': [60, 65, 70, 58, 75, 62, 68, 66, 72, 59],
        'is_person': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # All points labelled as 0
    }
    df = pd.DataFrame(data)

    # Split the data into features (X) and target variable (y)
    X = df[['height', 'weight']]
    y = df['is_person']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a logistic regression model
    model = LogisticRegression()

    # Train the model on the training data
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # Print the results
    print(f"Accuracy: {accuracy:.3f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

    # Return the final output (accuracy)
    return accuracy

# Call the method for validation
output = method()
print(f"Final Output: {output}")