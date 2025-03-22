import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1x1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 2: 1x1 convolution followed by a 3x3 convolution
    branch1x1_3x3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3x3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(branch1x1_3x3)

    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    branch1x1_3x3_3x3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3x3_3x3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(branch3x3_3x3)
    branch3x3_3x3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(branch3x3_3x3)

    # Concatenate the outputs from the three branches
    concatenated = layers.Concatenate()([branch1x1, branch3x3, branch3x3_3x3])

    # 1x1 convolution to adjust the output dimensions
    adjusted_dimensions = layers.Conv2D(64, (1, 1), activation='relu')(concatenated)

    # Flatten the output
    flattened = layers.Flatten()(adjusted_dimensions)

    # Fully connected layers
    fc1 = layers.Dense(128, activation='relu')(flattened)
    fc2 = layers.Dense(64, activation='relu')(fc1)
    outputs = layers.Dense(10, activation='softmax')(fc2)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()