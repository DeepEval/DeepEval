import numpy as np

def method(y_true, y_pred):
    y_mean_true = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_mean_true) ** 2)
    output = 1 - numerator / denominator
    return output

# Example usage:
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
output = method(y_true, y_pred)
print(output)