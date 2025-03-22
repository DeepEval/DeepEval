import tensorflow as tf

def method(input_data):
    # Apply embedding to the input data using TensorFlow
    embedded_data = tf.keras.layers.Embedding(input_dim=100, output_dim=100)(input_data)

    # Return the embedded sequence
    return embedded_data

# Test the method
input_data = tf.random.normal([1000, 100])
output = method(input_data)
print(output)