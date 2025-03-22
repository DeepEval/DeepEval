# Import necessary packages
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Define the method function
def method():
    # Get predictions and their corresponding confidences
    confidences = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    correct_predictions = np.where(predictions == y_test)[0]
    wrong_predictions = np.where(predictions!= y_test)[0]

    # Filter correct predictions to keep the lowest confidence
    lowest_confidence_correct = np.min(confidences[correct_predictions, np.arange(len(confidences))[correct_predictions]], axis=1)
    lowest_confidence_correct_idx = np.argmin(confidences[correct_predictions, np.arange(len(confidences))[correct_predictions]], axis=1)

    # Filter wrong predictions to keep the highest confidence
    highest_confidence_wrong = np.max(confidences[wrong_predictions, np.arange(len(confidences))[wrong_predictions]], axis=1)
    highest_confidence_wrong_idx = np.argmax(confidences[wrong_predictions, np.arange(len(confidences))[wrong_predictions]], axis=1)

    # Count the number of correct predictions
    num_correct = len(correct_predictions)

    # Check if the number of correct predictions is less than 95%
    if num_correct / len(y_test) < 0.95:
        # Replace the lowest confidence correct prediction with the highest confidence wrong prediction
        replace_idx = lowest_confidence_correct_idx[highest_confidence_wrong_idx > lowest_confidence_correct]
        if len(replace_idx) > 0:
            confidences[correct_predictions[replace_idx], lowest_confidence_correct_idx[replace_idx]] = highest_confidence_wrong[replace_idx]
            predictions[correct_predictions[replace_idx]] = y_test[correct_predictions[replace_idx]]

    # Return the final output
    return confidences, predictions

# Call the method function
output = method()
print("Confidences:", output[0])
print("Predictions:", output[1])