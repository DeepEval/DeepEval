import numpy as np

def method():
    data = [1, 2, 3, 4, 5]  
    mean = np.mean(data)
    absolute_deviations = np.abs(np.array(data) - mean)
    mad = np.mean(absolute_deviations)
    return mad

# Test Case
output = method()
print("Mean Absolute Deviation:", output)