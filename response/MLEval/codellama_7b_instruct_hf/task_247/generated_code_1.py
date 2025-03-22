import numpy as np

def method(X):
    # Compute the sum along the second and fourth elements
    output1 = np.sum(X[:, 1, :], axis=1)
    output2 = np.sum(X[:, 3, :], axis=1)

    # Compute the sum along the first and third elements
    output3 = np.sum(X[:, 0, :], axis=1)
    output4 = np.sum(X[:, 2, :], axis=1)

    # Concatenate the outputs
    output = np.concatenate((output1, output2, output3, output4), axis=1)

    return output

# Test the method
X = np.random.rand(4, 5, 6)
output = method(X)
print(output)