import numpy as np

def method(X):
    output = np.cumprod(X, axis=1)
    return output

# Test the method
X = np.array([[1, 2, 3], [4, 5, 6]])
print(method(X))