from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def method():
    model = Sequential()
    model.add(LSTM(128, input_shape=(timesteps, features)))  # Adjust timesteps and features based on your data
    model.add(Dropout(0.2))  # Add dropout to prevent overfitting
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))  # Add another dropout layer
    model.add(Dense(1))  # Output layer (adjust units based on your task)

    model.compile(loss='mean_squared_error', optimizer='adam')  # Adjust loss function and optimizer as needed

    return model

# Example usage:
model = method()
model.summary()  # Print model summary