import tensorflow as tf

def method():
  # Create a SparseTensor with some values
  values = [1, 2, 3, 4]
  indices = [[0, 0], [1, 1], [2, 2], [3, 3]]
  dense_shape = [4, 4]
  sparse_tensor = tf.SparseTensor(indices, values, dense_shape)

  # Convert the SparseTensor to Tensor
  tensor = tf.sparse.to_dense(sparse_tensor)

  # Print the Tensor
  print(tensor)

# Call the method
method()