import numpy as np
from scipy.special import softmax

# Define the fully connected layer
fc_layer = np.array([[1, 2, 3], [4, 5, 6]])

# Apply softmax to the fully connected layer
output = softmax(fc_layer)

print(output)