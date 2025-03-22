import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method():
    # Creating a shallow neural network
    model = Sequential()
    model.add(Dense(units=10, activation='relu', input_shape=(20,)))  # Input layer with 20 features and hidden layer with 10 neurons
    model.add(Dense(units=1, activation='sigmoid'))  # Output layer with 1 neuron for binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    # Example output (the model object itself can be returned)
    output = model
    return output

# Call the method for validation
output = method()