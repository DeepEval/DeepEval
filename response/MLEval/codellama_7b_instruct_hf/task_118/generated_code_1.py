import numpy as np

def method(tensor):
    # Calculate the gradient of the tensor
    gradient = np.gradient(tensor)

    # Return the gradient
    return gradient

# Test the method
tensor = np.array([[1, 2, 3], [4, 5, 6]])
output = method(tensor)
print(output)