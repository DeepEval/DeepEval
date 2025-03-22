import numpy as np

def method():
    # Define two vectors (arrays)
    vector_a = np.array([1, 2, 3])
    vector_b = np.array([4, 5, 6])
    
    # Perform an element-wise addition of the two vectors
    output = vector_a + vector_b
    
    return output

# Call the method for validation
result = method()
print("Output:", result)