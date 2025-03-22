import numpy as np

def method():
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    output = np.cumprod(X, axis=1)
    return output

output = method()
print(output)