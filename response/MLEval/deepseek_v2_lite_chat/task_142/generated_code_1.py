import numpy as np

def method():
    # Create a 2D array (matrix)
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    # Calculate the cumulative product along the second axis (columns)
    cumulative_product = np.cumprod(X, axis=0)  # axis=0 ensures column-wise operation
    
    # Return the result
    return cumulative_product

# Call the method for validation
output = method()
print(output)