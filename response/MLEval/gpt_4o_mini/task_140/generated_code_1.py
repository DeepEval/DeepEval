import numpy as np

def method():
    # Define two example arrays x and y
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    
    # Add x and y element-wise
    output = np.add(x, y)
    
    return output

# Call the method for validation
result = method()
print(result)