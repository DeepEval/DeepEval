import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model
    inputs = layers.Input(shape=input_shape)

    # First convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second convolutional layer
    y = layers.Conv2D(64, (3, 3), activation='relu')(x)
    y = layers.MaxPooling2D((2, 2))(y)

    # Third convolutional layer
    z = layers.Conv2D(128, (3, 3), activation='relu')(y)
    z = layers.MaxPooling2D((2, 2))(z)

    # Concatenate the outputs along the channel dimension
    concatenated = layers.Concatenate(axis=-1)([x, y, z])

    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)

    # First fully connected layer
    fc1 = layers.Dense(128, activation='relu')(flattened)
    fc1 = layers.Dropout(0.5)(fc1)

    # Second fully connected layer
    outputs = layers.Dense(10, activation='softmax')(fc1)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()