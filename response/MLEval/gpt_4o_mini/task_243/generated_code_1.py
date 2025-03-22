import numpy as np

def method():
    # Create a sample 3D array for demonstration purposes
    X = np.random.rand(4, 3, 2)  # Example shape (4, 3, 2)

    # Transpose the last two dimensions
    output = np.transpose(X, (0, 2, 1))  # Transpose the last two dimensions

    return output

# Call the method for validation
result = method()
print(result)