import numpy as np

def compute_loss(X, y, weights):
    """Compute the mean squared error loss."""
    predictions = X.dot(weights)
    loss = np.mean((predictions - y) ** 2)
    return loss

def l1_regularization(weights, lambda_reg):
    """Compute the L1 regularization term."""
    return lambda_reg * np.sum(np.abs(weights))

def method(X, y, weights, lambda_reg):
    """Compute the total loss with L1 regularization."""
    mse_loss = compute_loss(X, y, weights)
    l1_loss = l1_regularization(weights, lambda_reg)
    total_loss = mse_loss + l1_loss
    return total_loss

# Example usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(0)
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.rand(100)       # 100 target values
    weights = np.random.rand(10)  # Initialize weights
    lambda_reg = 0.1              # Regularization strength

    # Call the method
    output = method(X, y, weights, lambda_reg)
    print("Total loss with L1 regularization:", output)