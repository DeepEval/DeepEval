import numpy as np

def method():
    # Assuming X is a 3D numpy array
    X = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
    
    # Transpose the last two dimensions of X
    output = np.transpose(X, (2, 0, 1))
    
    return output

# Call the method for validation
result = method()
print(result)