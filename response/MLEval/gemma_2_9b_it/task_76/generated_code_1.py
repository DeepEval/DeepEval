import tensorflow as tf

def method():
  # Create a simple TensorFlow graph
  a = tf.constant(2)
  b = tf.constant(3)
  c = tf.add(a, b)

  # Create a TensorFlow SummaryWriter
  writer = tf.summary.FileWriter('logs')

  # Add the operation to the TensorBoard summary
  tf.summary.scalar('sum', c)

  # Write the graph to TensorBoard
  writer.add_graph(tf.get_default_graph())

  # Close the SummaryWriter
  writer.close()

  return "TensorBoard summary written to logs directory"

# Call the method and print the output
output = method()
print(output)