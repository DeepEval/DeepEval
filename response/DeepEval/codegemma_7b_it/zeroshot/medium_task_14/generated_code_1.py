import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():

    # Input layer
    input_img = layers.Input(shape=(32, 32, 3))

    # Block 1
    x1 = layers.Conv2D(64, (3, 3), padding='same')(input_img)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    # Block 2
    x2 = layers.Conv2D(64, (3, 3), padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    # Block 3
    x3 = layers.Conv2D(64, (3, 3), padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    # Parallel branch
    parallel_x = layers.Conv2D(64, (3, 3), padding='same')(input_img)
    parallel_x = layers.BatchNormalization()(parallel_x)
    parallel_x = layers.Activation('relu')(parallel_x)

    # Output paths
    output_x1 = layers.Conv2D(10, (1, 1), padding='same')(x1)
    output_x2 = layers.Conv2D(10, (1, 1), padding='same')(x2)
    output_x3 = layers.Conv2D(10, (1, 1), padding='same')(x3)
    parallel_output = layers.Conv2D(10, (1, 1), padding='same')(parallel_x)

    # Add outputs
    aggregated_output = layers.add([output_x1, output_x2, output_x3, parallel_output])

    # Fully connected layers
    flattened_output = layers.Flatten()(aggregated_output)
    output = layers.Dense(10, activation='softmax')(flattened_output)

    # Model creation
    model = models.Model(inputs=input_img, outputs=output)

    return model