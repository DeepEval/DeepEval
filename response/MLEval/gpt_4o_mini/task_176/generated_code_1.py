import tensorflow as tf

def method():
    # Example logits
    logits = tf.constant([2.0, 1.0, 0.1])
    
    # Compute softmax activations
    output = tf.nn.softmax(logits)
    
    return output

# Call the method for validation
output_result = method()
print(output_result.numpy())  # Convert the tensor to a numpy array for easier viewing