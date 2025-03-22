import numpy as np

def method(y_true, y_pred):
    # Convert inputs to numpy arrays to ensure compatibility with numpy operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the mean of the true values
    y_mean_true = np.mean(y_true)
    
    # Calculate the total sum of squares (SST)
    sst = np.sum((y_true - y_mean_true) ** 2)
    
    # Calculate the sum of squares of residuals (SSR)
    ssr = np.sum((y_true - y_pred) ** 2)
    
    # Calculate the R-squared value
    r_squared = 1 - (ssr / sst)
    
    return r_squared

# Example usage
y_true_example = [3, -0.5, 2, 7]
y_pred_example = [2.5, 0.0, 2, 8]

output = method(y_true_example, y_pred_example)
print("R-squared:", output)