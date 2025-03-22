import numpy as np

def method(x, y):
    return np.remainder(x, y)

# Example usage
x = np.array([10, 20, 30, 40])
y = np.array([2, 4, 6, 8])
output = method(x, y)

# Validate output
print(output)