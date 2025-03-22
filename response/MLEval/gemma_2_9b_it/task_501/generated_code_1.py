import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def method():
    model = Sequential()
    model.add(LSTM(128, input_shape=(timesteps, features)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))  # Output layer with 1 neuron for regression

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Example usage:
timesteps = 10  # Adjust based on your data
features = 5    # Adjust based on your data
model = method()
model.summary()