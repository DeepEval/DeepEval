import tensorflow as tf

def method():
  # Create a simple TensorFlow graph
  a = tf.constant(3.0, name="a")
  b = tf.constant(4.0, name="b")
  c = tf.add(a, b, name="add")

  # Create a SummaryWriter object
  writer = tf.summary.create_file_writer("logs")

  # Add a scalar summary for the addition operation
  with writer.as_default():
    tf.summary.scalar("addition", c, step=0)

  # Close the SummaryWriter object
  writer.close()

  # Call the method for validation
  method()