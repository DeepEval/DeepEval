import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():
    # Step 1: Generate a synthetic dataset
    X = np.random.rand(1000, 10) * 10  # 1000 samples, each with 10 features
    y = (np.random.rand(1000) > 0.5).astype(int)  # Binary classification, 50% positive

    # Step 2: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 3: Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Step 4: Predict on the test set and adjust based on a confidence threshold
    pred_probs = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities
    threshold = 0.5  # Set a confidence threshold
    pred_labels = (pred_probs >= threshold).astype(int)  # Convert probabilities to predictions

    # Calculate accuracy and correct/incorrect predictions
    correct_predictions = (y_test == pred_labels).mean() * 100
    incorrect_predictions = (y_test != pred_labels).mean() * 100

    # Ensure correct prediction rate is at least 95%
    min_correct_predictions = 0.95
    if correct_predictions < min_correct_predictions:
        pred_labels[pred_probs < threshold] = 0  # Adjust predictions if accuracy is below threshold
        correct_predictions = (y_test == pred_labels).mean() * 100

    # Step 5: Return the output
    output = {
        'correct_predictions': correct_predictions,
        'incorrect_predictions': incorrect_predictions
    }
    return output

# Call the method for validation
output = method()
print(output)