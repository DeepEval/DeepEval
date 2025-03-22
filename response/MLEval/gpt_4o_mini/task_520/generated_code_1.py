import tensorflow as tf

def method(sparse_tensor):
    # Convert the SparseTensor to a dense Tensor
    dense_tensor = tf.sparse.to_dense(sparse_tensor)
    return dense_tensor

# Example usage for validation
if __name__ == "__main__":
    # Create a SparseTensor
    indices = [[0, 0], [1, 2], [2, 3]]
    values = [1, 2, 3]
    dense_shape = [3, 4]
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    # Call the method to convert SparseTensor to Tensor
    output = method(sparse_tensor)

    # Print the output
    print("Dense Tensor:\n", output.numpy())