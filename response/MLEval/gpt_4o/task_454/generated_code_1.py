# Import necessary packages
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method():
    # Define a shallow neural network
    model = Sequential()

    # Input layer and one hidden layer (e.g., with 10 units and ReLU activation)
    model.add(Dense(units=10, activation='relu', input_shape=(5,)))  # Assuming input features of size 5

    # Output layer (e.g., with 1 unit for binary classification)
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Call the method for validation
network_model = method()
print(network_model.summary())