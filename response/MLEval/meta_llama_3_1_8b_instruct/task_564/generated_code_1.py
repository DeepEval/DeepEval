# Import necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def method():
    """
    This function creates a recurrent layer composed of LSTM cells.
    
    Returns:
    output (tf.keras.Model): The output of the recurrent layer.
    """
    
    # Define the input shape
    input_shape = (10, 5)  # (time_steps, features)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Create the LSTM layer
    lstm_layer = layers.LSTM(64, return_sequences=True, activation='relu')
    
    # Create the recurrent layer composed of LSTM cells
    recurrent_layer = lstm_layer(inputs)
    
    # Create the output layer
    outputs = layers.Dense(64, activation='relu')(recurrent_layer)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Return the output of the model
    return model

# Call the generated method for validation
output = method()
print(output.summary())