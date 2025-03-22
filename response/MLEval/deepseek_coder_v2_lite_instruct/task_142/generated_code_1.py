import numpy as np

def method():
    # Generate a sample 2D array (matrix)
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Calculate the cumulative product along the second axis
    cumulative_product = np.cumprod(X, axis=1)
    
    # Return the final output
    return cumulative_product

# Call the method for validation
output = method()
print(output)