import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def method():
    # Set the encoding dimension
    ENCODING_DIM = 64  # You can change this to your desired size

    # Create a sequential model
    model = Sequential()

    # Input layer: Flatten the input images (28x28)
    model.add(Flatten(input_shape=(28, 28)))

    # Hidden layer: Add a dense layer with ReLU activation
    model.add(Dense(128, activation='relu'))

    # Output layer: Add a dense layer for encoding to the specified dimension
    model.add(Dense(ENCODING_DIM))

    # Note: We do not compile the model as per the requirement
    output = model

    return output

# Call the method for validation
encoder_model = method()
print(encoder_model.summary())