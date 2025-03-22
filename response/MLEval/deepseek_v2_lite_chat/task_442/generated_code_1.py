import numpy as np

def method():
    # Example binary classification dataset (hypothetical data, replace with actual data)
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    y_true = np.array([
        [0],
        [1],
        [1]
    ])

    # Convert y_true to binary (0 or 1) to match the threshold
    y_true_binary = (y_true >= 0.5)

    # Initialize model parameters (for simplicity, assume we have trained model parameters)
    w = np.random.randn(X.shape[1])  # weights
    b = 0  # bias

    # Predict function
    def predict(x):
        return np.sign(np.dot(x, w) + b)

    # Evaluate the model
    correct = 0
    total = len(y_true)
    for i in range(total):
        pred = predict(X[i])
        if pred == y_true_binary[i][0]:
            correct += 1

    accuracy = correct / total
    return f"Model Accuracy: {accuracy:.2f}"

# Call the method for validation
print(method())