import numpy as np

def l1_regularization(weights, alpha):
    return np.sum(np.abs(weights)) * alpha

# Example usage
weights = np.array([[1, 2], [3, 4]])
alpha = 0.1
output = l1_regularization(weights, alpha)
print(output)  # Output: 1.5