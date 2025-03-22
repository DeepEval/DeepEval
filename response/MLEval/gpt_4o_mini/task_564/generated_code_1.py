import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def method():
    # Define the model
    model = Sequential()
    
    # Add an LSTM layer
    model.add(LSTM(64, input_shape=(None, 1)))  # 64 LSTM units, expecting input of shape (timesteps, features)

    # Add a Dense layer for output
    model.add(Dense(1))  # Assuming a single output

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Create some dummy data for validation
    X_train = np.random.rand(100, 10, 1)  # 100 samples, 10 timesteps, 1 feature
    y_train = np.random.rand(100, 1)      # 100 target values
    
    # Fit the model to the dummy data
    model.fit(X_train, y_train, epochs=5, batch_size=16)

    # Return the model summary as output
    output = model.summary()
    
    return output

# Call the method for validation
method()