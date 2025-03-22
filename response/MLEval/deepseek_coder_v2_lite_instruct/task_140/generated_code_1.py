import numpy as np

def method():
    # Define two arrays x and y
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    
    # Perform element-wise addition
    output = x + y
    
    return output

# Call the method and print the result for validation
result = method()
print(result)  # Output should be [5, 7, 9]