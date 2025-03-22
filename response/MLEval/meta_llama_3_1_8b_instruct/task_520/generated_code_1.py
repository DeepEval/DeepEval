# Import necessary packages
import tensorflow as tf
import numpy as np

# Define the custom function to convert SparseTensor to Tensor
def method():
    # Create a SparseTensor
    indices = [[0, 0], [1, 2], [2, 0]]
    values = [1.0, 2.0, 3.0]
    dense_shape = [3, 3]
    sparse_tensor = tf.SparseTensor(indices, values, dense_shape)

    # Convert SparseTensor to Tensor
    tensor = tf.sparse.to_dense(sparse_tensor)

    return tensor

# Call the method() function for validation
output = method()
print(output)

# Alternatively, you can use NumPy to achieve the same result
def method_numpy():
    # Create a SparseTensor
    indices = [[0, 0], [1, 2], [2, 0]]
    values = [1.0, 2.0, 3.0]
    dense_shape = [3, 3]
    sparse_tensor = tf.SparseTensor(indices, values, dense_shape)

    # Convert SparseTensor to Tensor using NumPy
    tensor = tf.sparse.to_dense(sparse_tensor).numpy()

    return tensor

# Call the method_numpy() function for validation
output_numpy = method_numpy()
print(output_numpy)