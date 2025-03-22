import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def method():
    # Define the model
    model = Sequential()

    # Input layer and first hidden layer with, for example, 10 nodes
    model.add(Dense(10, activation='relu', input_shape=(input_dim,)))  # Replace input_dim with the number of features

    # Additional hidden layer with 6 nodes
    model.add(Dense(6, activation='relu'))

    # Output layer (for a binary classification example)
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    return model

# Call the method to validate
output_model = method()