import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    # Concatenate input with conv1 output
    concat1 = layers.Concatenate(axis=-1)([input_layer, conv1])

    # Second convolutional layer
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    # Concatenate previous output with conv2 output
    concat2 = layers.Concatenate(axis=-1)([concat1, conv2])

    # Third convolutional layer
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat2)
    # Concatenate previous output with conv3 output
    concat3 = layers.Concatenate(axis=-1)([concat2, conv3])

    # Flatten the output
    flatten = layers.Flatten()(concat3)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dense2 = layers.Dense(10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=input_layer, outputs=dense2)

    return model

# Example usage:
model = dl_model()
model.summary()