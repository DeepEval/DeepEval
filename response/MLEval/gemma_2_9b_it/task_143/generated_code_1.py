import numpy as np

def method():
    x = np.array([10, 20, 30, 40, 50])
    y = np.array([3, 5, 7, 11, 13])
    output = np.mod(x, y)
    return output

# Validation
print(method())