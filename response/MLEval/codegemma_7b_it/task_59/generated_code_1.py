import numpy as np

def l1_regularization(weights, lambda_param):
    """
    Implements L1 regularization for linear regression.

    Args:
        weights: The weights vector for the linear model.
        lambda_param: The regularization parameter (lambda).

    Returns:
        The regularized loss value.
    """

    # Calculate the L1 norm of the weights
    l1_norm = np.sum(np.abs(weights))

    # Calculate the loss function with L1 regularization
    loss = np.mean((y_true - np.dot(X, weights))**2) + (lambda_param / 2) * l1_norm

    return loss

# Example usage:
# X: Input data (features)
# y_true: True labels
# weights: Initial weights vector
# lambda_param: Regularization parameter

loss = l1_regularization(weights, lambda_param)

print("L1 Regularized Loss:", loss)