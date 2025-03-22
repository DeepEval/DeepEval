import numpy as np

def method():
    y_true = np.array([100, 95, 105, 100, 90, 95, 110, 100, 115, 105])
    y_pred = np.array([102, 98, 107, 101, 92, 98, 112, 102, 110, 105])

    # Calculate the mean of true values
    y_mean_true = np.mean(y_true)

    # Calculate the sum of squared differences between true and predicted values
    squared_diff = (y_true - y_pred) ** 2

    # Calculate the sum of squared differences between true and mean of true values
    squared_diff_from_mean = (y_true - y_mean_true) ** 2

    # Calculate the coefficient of determination
    r2 = 1 - (np.sum(squared_diff) / np.sum(squared_diff_from_mean))

    return r2

# Call the method for validation
output = method()

# Print the output
print(output)