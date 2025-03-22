import tensorflow as tf

def method():
    # Create a SparseTensor for demonstration purposes
    indices = [[0, 0], [1, 2], [2, 3]]
    values = [1, 2, 3]
    dense_shape = [3, 4]

    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    # Convert SparseTensor to a dense Tensor
    dense_tensor = tf.sparse.to_dense(sparse_tensor)

    # Return the dense Tensor
    return dense_tensor

# Call the method function and validate the output
output = method()
print("Dense Tensor:")
print(output)