import tensorflow as tf

def method():
    # Define logits
    logits = tf.constant([[1.0, 2.0, 3.0]])

    # Calculate softmax activations
    softmax_activations = tf.nn.softmax(logits)

    # Return softmax activations
    return softmax_activations

# Call the method to validate
output = method()

# Print the output
print(output)