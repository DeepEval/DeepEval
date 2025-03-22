import numpy as np

def method(y_true, y_pred):
    # Convert the input to numpy arrays for vectorized operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the mean of the true values
    y_mean_true = np.mean(y_true)

    # Calculate the total sum of squares
    ss_total = np.sum((y_true - y_mean_true) ** 2)

    # Calculate the residual sum of squares
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # Calculate the coefficient of determination (R²)
    r_squared = 1 - (ss_residual / ss_total)

    return r_squared

# Example usage for validation
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

output = method(y_true, y_pred)
print("Coefficient of Determination (R²):", output)