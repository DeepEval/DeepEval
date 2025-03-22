import numpy as np

def l1_regularization(weights, lambda_):
    """
    Compute the L1 regularization term.

    Parameters:
    - weights: numpy array, model weights
    - lambda_: float, regularization strength

    Returns:
    - l1_penalty: float, L1 regularization penalty
    """
    l1_penalty = lambda_ * np.sum(np.abs(weights))
    return l1_penalty

def method(X, y, weights, lambda_):
    """
    Compute the L1 regularized loss for a simple linear regression model.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), input features
    - y: numpy array of shape (n_samples,), target values
    - weights: numpy array of shape (n_features,), model weights
    - lambda_: float, regularization strength

    Returns:
    - total_loss: float, total loss including L1 regularization
    """
    # Compute predictions
    predictions = X.dot(weights)
    
    # Compute mean squared error loss
    mse_loss = np.mean((y - predictions) ** 2)
    
    # Compute L1 regularization penalty
    l1_penalty = l1_regularization(weights, lambda_)
    
    # Total loss is the sum of MSE loss and L1 penalty
    total_loss = mse_loss + l1_penalty

    return total_loss

# Example validation
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    weights = np.array([0.5, -0.5])
    lambda_ = 0.1

    # Calculate the regularized loss
    total_loss = method(X, y, weights, lambda_)
    print("Total Loss with L1 Regularization:", total_loss)