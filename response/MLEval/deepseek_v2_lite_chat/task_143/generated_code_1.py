import numpy as np

def method():
    # Input arrays
    x = np.array([1, 2, 3, 4])
    y = np.array([0.5, 1.0, 1.5, 2.0])
    
    # Compute element-wise remainder
    remainder = x % y
    
    return remainder

# Validate the method
output = method()
print("Output:", output)