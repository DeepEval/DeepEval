import numpy as np

def method():
    # Example input arrays x and y
    x = np.array([10, 20, 30, 40, 50])
    y = np.array([3, 7, 5, 8, 6])
    
    # Compute the remainder of x / y element-wise
    output = np.remainder(x, y)
    
    return output

# Call the method for validation
result = method()
print(result)