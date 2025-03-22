import numpy as np

def method(x, y):
    return np.add(x, y)

# Test the method
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
output = method(x, y)
print(output)  # Output: [5, 7, 9]