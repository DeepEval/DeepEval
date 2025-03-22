# Import necessary packages
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Main path: increase the feature map width and then restore the number of channels
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(1, (1, 1), activation='relu')(x)

    # Branch path: directly connect to the input
    branch = layers.Conv2D(32, (3, 3), activation='relu')(inputs)

    # Combine the two paths through an addition operation
    x = layers.Add()([x, branch])

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create and print the model
model = dl_model()
print(model.summary())