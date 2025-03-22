import tensorflow as tf
import numpy as np

def method():
    # Generate a random tensor of logits for demonstration
    logits = np.random.rand(3)  # 3 is just an example. In practice, logits are usually calculated from your model.
    
    # Create a TensorFlow tensor from the NumPy array
    logits_tensor = tf.convert_to_tensor(logits)
    
    # Apply softmax to the logits
    softmax = tf.nn.softmax(logits_tensor)
    
    return softmax

# Call the method for validation
output = method()
print("Output:", output)