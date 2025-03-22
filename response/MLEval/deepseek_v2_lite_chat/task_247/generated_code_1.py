import numpy as np

def method():
    # Example array (8 elements for demonstration)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    
    # Compute sum along second and fourth elements
    sum1 = X[[1, 3]].sum()
    
    # Compute sum along first and third elements
    sum2 = X[[0, 2]].sum()
    
    # Combine the results
    output = sum1 + sum2
    
    return output

# Call the method for validation
result = method()
print("Result:", result)