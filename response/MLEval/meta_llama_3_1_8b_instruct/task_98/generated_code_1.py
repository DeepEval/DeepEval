import tensorflow as tf
import numpy as np


def method():
    """
    Creates a TensorFlow placeholder with a vector shape.

    Returns:
        output: The output of the method.
    """

    # Create a placeholder with a vector shape
    placeholder = tf.placeholder(tf.float32, shape=(3,))

    # Create a vector with three elements
    vector = np.array([1.0, 2.0, 3.0])

    # Use the placeholder to create a tensor
    tensor = tf.cast(placeholder, tf.float32)

    # Return the output of the method
    return tensor.eval(feed_dict={placeholder: vector})


# Call the method and store the output
output = method()

# Print the output
print(output)