import tensorflow as tf

def method():
    # Example SparseTensor
    indices = [[0, 0], [1, 2], [2, 3]]
    values = [1, 2, 3]
    dense_shape = [3, 4]
    
    sparse_tensor = tf.sparse.SparseTensor(indices, values, dense_shape)
    
    # Convert SparseTensor to Tensor
    output = tf.sparse.to_dense(sparse_tensor)
    
    return output

# Call the method for validation
output = method()
print(output)