import tensorflow as tf

def method(logits):
    # Apply softmax to the input logits
    output = tf.nn.softmax(logits)
    return output

# Example validation
if __name__ == "__main__":
    # Example logits
    logits = [2.0, 1.0, 0.1]
    # Call the method and print the output
    softmax_output = method(logits)
    print("Softmax Output:", softmax_output.numpy())