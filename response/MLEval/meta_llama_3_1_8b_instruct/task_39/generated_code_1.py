import numpy as np

def method(y_true, y_pred):
    """
    Calculate the coefficient of determination (R-squared) using the given formula.

    Args:
    y_true (array-like): The true values.
    y_pred (array-like): The predicted values.

    Returns:
    float: The coefficient of determination (R-squared).
    """
    # Calculate the mean of the true values
    y_mean_true = np.mean(y_true)

    # Calculate the sum of the squared differences between the true and predicted values
    sum_squared_diff = np.sum((y_true - y_pred) ** 2)

    # Calculate the sum of the squared differences between the true and mean true values
    sum_squared_diff_mean = np.sum((y_true - y_mean_true) ** 2)

    # Calculate the coefficient of determination (R-squared)
    r_squared = 1 - (sum_squared_diff / sum_squared_diff_mean)

    return r_squared

# Generate sample data for validation
np.random.seed(0)
y_true = np.random.rand(100)
y_pred = np.random.rand(100)

# Call the method for validation
r_squared = method(y_true, y_pred)
print("Coefficient of Determination (R-squared):", r_squared)