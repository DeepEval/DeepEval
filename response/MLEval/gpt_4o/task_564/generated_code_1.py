import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def method():
    # Define the input shape
    input_shape = (10, 1)  # 10 timesteps with 1 feature each, for example

    # Initialize the Sequential model
    model = Sequential()

    # Add an LSTM layer with a specified number of units
    model.add(LSTM(units=50, input_shape=input_shape, return_sequences=False))

    # Add a Dense layer to produce the final output
    model.add(Dense(units=1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Print the model summary
    model.summary()

    return model

# Call the method for validation
model = method()