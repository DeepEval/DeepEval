import tensorflow as tf

def method():
    # Define the LSTM layer
    lstm_layer = tf.keras.layers.LSTM(units=64, return_sequences=True)

    # Example input sequence
    inputs = tf.random.normal(shape=(32, 10, 128))  # (batch_size, sequence_length, input_dim)

    # Apply the LSTM layer
    outputs = lstm_layer(inputs)

    # Return the final output
    return outputs

# Call the method for validation
output = method()
print(output)