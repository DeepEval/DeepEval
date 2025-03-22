import numpy as np

def method(X):
    # Transpose the last two dimensions of X
    output = np.transpose(X, (0, 2, 1))
    return output

# Test the method
X = np.random.rand(3, 4, 5)
print(method(X))


import numpy as np

def method(X):
    # Transpose the last two dimensions of X
    output = np.swapaxes(X, -1, -2)
    return output

# Test the method
X = np.random.rand(3, 4, 5)
print(method(X))