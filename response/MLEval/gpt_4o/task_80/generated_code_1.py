# Import necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
import numpy as np

def method():
    # Create a sequential model
    model = Sequential()

    # Add a dense layer
    model.add(Dense(units=64, input_shape=(100,)))

    # Insert a Batch Normalization layer
    model.add(BatchNormalization())

    # Add another Dense layer
    model.add(Dense(units=32))

    # Insert another Batch Normalization layer
    model.add(BatchNormalization())

    # Add an output Dense layer
    model.add(Dense(units=10))

    # Optionally, add an activation function to the output
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Call the method for validation
model = method()
print(model.summary())