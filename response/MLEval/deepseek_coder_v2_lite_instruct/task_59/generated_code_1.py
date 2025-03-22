import numpy as np

def method():
    # Example data
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Features
    y = np.array([1, 2, 3])  # Targets
    theta = np.array([0.1, 0.2])  # Initial weights

    # Hyperparameters
    alpha = 0.01  # Regularization strength

    # Cost function with L1 regularization
    def cost_function(X, y, theta, alpha):
        m = len(y)
        predictions = X.dot(theta)
        error = predictions - y
        cost = (1 / (2 * m)) * np.sum(error ** 2) + (alpha * np.sum(np.abs(theta)))
        return cost

    # Example call to the cost function
    output = cost_function(X, y, theta, alpha)
    return output

# Call the method for validation
output = method()
print(output)