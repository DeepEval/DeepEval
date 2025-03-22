import numpy as np

def method():
    # Define two arrays
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    
    # Add x and y element-wise
    result = x + y
    
    # Return the result
    return result

# Call the method for validation
output = method()
print("Output:", output)