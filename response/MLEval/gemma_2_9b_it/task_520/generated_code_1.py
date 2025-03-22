import tensorflow as tf

def method():
  # Create a SparseTensor
  indices = tf.constant([[0, 0], [1, 1], [2, 2]])
  values = tf.constant([1, 2, 3])
  shape = tf.constant([3, 3])
  sparse_tensor = tf.sparse.SparseTensor(indices, values, shape)

  # Convert SparseTensor to Tensor
  tensor = tf.sparse.to_dense(sparse_tensor)

  return tensor

# Call the method and print the output
output = method()
print(output)