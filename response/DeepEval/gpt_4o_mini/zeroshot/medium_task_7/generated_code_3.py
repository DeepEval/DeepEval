import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)

    # Second convolutional layer
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(conv1)

    # Third convolutional layer
    conv3 = layers.Conv2D(128, (3, 3), activation='relu')(conv2)

    # Direct path convolutional layer
    direct_conv = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Adding outputs of the first two convolutional layers with the third
    added = layers.add([conv2, conv3, direct_conv])

    # Flatten the output before fully connected layers
    flattened = layers.Flatten()(added)

    # First fully connected layer
    fc1 = layers.Dense(128, activation='relu')(flattened)

    # Second fully connected layer
    fc2 = layers.Dense(64, activation='relu')(fc1)

    # Output layer for classification (10 classes for CIFAR-10)
    outputs = layers.Dense(10, activation='softmax')(fc2)

    # Constructing the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()  # Print the model summary to verify the architecture