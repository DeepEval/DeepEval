import tensorflow as tf

def method():
    # Define the LSTM layer
    lstm_layer = tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(None, 10)) 

    return lstm_layer

# Call the method to get the LSTM layer
lstm_layer = method()

# Print the summary of the LSTM layer
lstm_layer.summary()