import numpy as np

def method():
    # Create a sample input array
    X = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])

    # Transpose the last two dimensions of X
    output = np.transpose(X, axes=[0, 2, 1])

    # Return the output
    return output

# Call the method for validation
output = method()

# Print the output
print(output)