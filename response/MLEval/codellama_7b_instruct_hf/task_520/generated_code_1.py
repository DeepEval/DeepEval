import tensorflow as tf

def convert_sparse_to_dense(sparse_tensor):
    return tf.sparse.to_dense(sparse_tensor)

# Example usage
sparse_tensor = tf.SparseTensor(indices=[[0, 0], [1, 1], [2, 2]], values=[1, 2, 3], dense_shape=[3, 3])
dense_tensor = convert_sparse_to_dense(sparse_tensor)
print(dense_tensor)