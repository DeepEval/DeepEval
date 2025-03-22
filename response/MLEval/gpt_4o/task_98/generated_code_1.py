import numpy as np

def method():
    # Creating vectors (or arrays) to act as placeholders
    vector_a = np.array([1, 2, 3])
    vector_b = np.array([4, 5, 6])

    # Performing element-wise addition
    output = np.add(vector_a, vector_b)
    
    return output

# Call the method for validation
result = method()
print("Output:", result)