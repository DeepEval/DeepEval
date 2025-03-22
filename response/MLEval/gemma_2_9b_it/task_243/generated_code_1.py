import numpy as np

def method():
    X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    output = np.transpose(X, axes=(0, 2, 1))
    return output

output = method()
print(output)