import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 channels (RGB)
    inputs = layers.Input(shape=input_shape)

    # First convolutional layer
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)

    # Second convolutional layer
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv2 = layers.BatchNormalization()(conv2)

    # Third convolutional layer
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv3 = layers.BatchNormalization()(conv3)

    # Direct convolutional layer processing the input
    direct_conv = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    direct_conv = layers.BatchNormalization()(direct_conv)

    # Add the outputs of the convolutional layers
    merged = layers.add([conv1, conv2, conv3, direct_conv])

    # Flatten the merged outputs
    flatten = layers.Flatten()(merged)

    # Fully connected layers
    dense1 = layers.Dense(256, activation='relu')(flatten)
    dense2 = layers.Dense(128, activation='relu')(dense1)

    # Output layer for classification (CIFAR-10 has 10 classes)
    outputs = layers.Dense(10, activation='softmax')(dense2)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # Optional: to display the model architecture