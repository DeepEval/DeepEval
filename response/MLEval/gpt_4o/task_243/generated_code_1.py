import numpy as np

def method(X):
    # Transpose the last two dimensions of X
    output = np.swapaxes(X, -1, -2)
    return output

# Example validation
if __name__ == "__main__":
    # Example input array
    X = np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]
    ])
    
    # Call the method and print the result
    result = method(X)
    print("Original array:\n", X)
    print("Transposed array:\n", result)