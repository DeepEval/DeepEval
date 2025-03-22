import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Initial convolution
    initial_conv = layers.Conv2D(32, (3, 3), padding='same')(input_layer)
    initial_conv = layers.BatchNormalization()(initial_conv)
    initial_conv = layers.ReLU()(initial_conv)

    # Parallel blocks
    block1 = layers.Conv2D(32, (3, 3), padding='same')(initial_conv)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.ReLU()(block1)

    block2 = layers.Conv2D(32, (3, 3), padding='same')(initial_conv)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.ReLU()(block2)

    block3 = layers.Conv2D(32, (3, 3), padding='same')(initial_conv)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.ReLU()(block3)

    # Combine outputs of parallel blocks
    merged = layers.add([block1, block2, block3])

    # Add the initial convolution's output to the merged output
    combined_output = layers.add([merged, initial_conv])

    # Flatten the output
    flattened = layers.Flatten()(combined_output)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense1 = layers.BatchNormalization()(dense1)

    dense2 = layers.Dense(64, activation='relu')(dense1)
    dense2 = layers.BatchNormalization()(dense2)

    # Output layer with softmax activation for classification
    output_layer = layers.Dense(10, activation='softmax')(dense2)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of model creation
model = dl_model()
model.summary()