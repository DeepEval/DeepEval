# Import necessary packages
import tensorflow as tf
import numpy as np

# Define a function to create a tensor with shape (1, 3, 3)
def method():
    # Create a tensor with shape (1, 3, 3)
    output = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], shape=(1, 3, 3))
    
    return output

# Call the method for validation
output = method()

# Print the shape of the output tensor
print(output.shape)

# Print the values of the output tensor
print(output.numpy())