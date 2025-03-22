import numpy as np

def method():
    # Example input array X
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    # Calculate the cumulative product along the second axis (axis=1)
    output = np.cumprod(X, axis=1)
    
    return output

# Call the method for validation
result = method()
print(result)