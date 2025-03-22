# Import necessary packages
import numpy as np

def method():
    # Define the input array X with shape (3, 4, 5)
    X = np.random.rand(3, 4, 5)
    
    # Transpose the last two dimensions of X
    output = np.transpose(X, (0, 2, 1))
    
    return output

# Call the generated method for validation
output = method()
print(output.shape)