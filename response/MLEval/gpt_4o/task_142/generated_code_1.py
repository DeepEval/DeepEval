import numpy as np

def method(X):
    # Calculate the cumulative product along the second axis (axis=1)
    output = np.cumprod(X, axis=1)
    return output

# Example validation
if __name__ == "__main__":
    # Example input
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    # Call the method
    result = method(X)
    
    # Print the result
    print("Cumulative product along the second axis:")
    print(result)